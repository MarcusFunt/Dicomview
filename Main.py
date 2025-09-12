#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modernized PyQt6 DICOM/Image Viewer
- Multi-series browser (select series from folder).
- Supports JPEG2000 (lossless & irreversible) via pylibjpeg-openjpeg or GDCM.
- Dark theme, modern split layout, toolbar.
- Zoom (wheel), pan (drag), fit, slice slider, drag-and-drop.
"""

import os, sys, math, traceback
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image
import pydicom
from pydicom.filereader import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut
from pydicom.pixel_data_handlers import pylibjpeg_handler, gdcm_handler

from PyQt6.QtCore import Qt, QRectF, QSize
from PyQt6.QtGui import (
    QAction,
    QImage,
    QPixmap,
    QKeySequence,
    QPalette,
    QColor,
    QTransform,
    QPainter,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QToolBar, QLabel, QSlider, QWidget, QVBoxLayout,
    QMessageBox, QStatusBar, QListWidget, QListWidgetItem, QComboBox, QTabWidget
)


# ---------------------------------------------------------------------
# JPEG2000 detection
# ---------------------------------------------------------------------
def jpeg2000_support_status() -> Tuple[bool, str]:
    details, ok = [], False
    try:
        if pylibjpeg_handler.is_available():
            ok = True; details.append("pylibjpeg-openjpeg: OK")
    except Exception as e:
        details.append(f"pylibjpeg handler error: {e}")
    try:
        if gdcm_handler.is_available():
            ok = True; details.append("gdcm: OK")
    except Exception as e:
        details.append(f"gdcm handler error: {e}")
    return ok, " | ".join(details) if details else "no details"


# ---------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------
def is_dicom_file(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            pre = f.read(132)
            if len(pre) >= 132 and pre[128:132] == b"DICM":
                return True
    except Exception:
        pass
    try:
        ds = dcmread(path, stop_before_pixels=True, force=True)
        return hasattr(ds, "SOPClassUID") or hasattr(ds, "PatientID")
    except Exception:
        return False


def try_open_image_any(path: str) -> Optional[Image.Image]:
    try:
        im = Image.open(path)
        im.load()
        return im
    except Exception:
        return None


def series_key_from_ds(ds: pydicom.dataset.FileDataset) -> str:
    return getattr(ds, "SeriesInstanceUID", None) or (
        f"{getattr(ds,'StudyInstanceUID','')}"
        f"|{getattr(ds,'PatientID','')}|{getattr(ds,'Modality','')}"
    )


def sort_key_from_ds(ds: pydicom.dataset.FileDataset) -> Tuple:
    ipp = getattr(ds, "ImagePositionPatient", None)
    z = float(ipp[2]) if isinstance(ipp, (list, tuple)) and len(ipp) >= 3 else math.inf
    return (z, getattr(ds, "InstanceNumber", 1e12))


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8: return arr
    a_min, a_max = np.nanmin(arr), np.nanmax(arr)
    if a_max == a_min: return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - a_min) / (a_max - a_min)
    return np.clip((scaled * 255).round(), 0, 255).astype(np.uint8)


def dicom_to_ndarray(ds: pydicom.dataset.FileDataset, frame_index: int = 0) -> np.ndarray:
    arr = ds.pixel_array
    if arr.ndim == 3 and getattr(ds,"NumberOfFrames",None):
        arr = arr[frame_index]
    elif arr.ndim == 4:
        arr = arr[frame_index]

    try: arr = apply_voi_lut(arr, ds)
    except Exception: pass

    slope, intercept = float(getattr(ds,"RescaleSlope",1.0)), float(getattr(ds,"RescaleIntercept",0.0))
    if slope != 1.0 or intercept != 0.0: arr = arr.astype(np.float32)*slope+intercept

    photometric = getattr(ds,"PhotometricInterpretation","MONOCHROME2").upper()
    if arr.ndim == 2:
        arr = normalize_to_uint8(arr)
        if photometric == "MONOCHROME1": arr = 255-arr
        return arr
    elif arr.ndim == 3 and arr.shape[2] == 3:
        return normalize_to_uint8(arr)
    return normalize_to_uint8(arr.squeeze())


def numpy_to_qimage(arr: np.ndarray) -> QImage:
    """Convert a numpy array to a QImage ensuring a contiguous buffer."""
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        h, w = arr.shape
        qimg = QImage(arr.tobytes(), w, h, arr.strides[0], QImage.Format.Format_Grayscale8)
        return qimg.copy()
    elif arr.ndim == 3 and arr.shape[2] == 3:
        h, w, _ = arr.shape
        qimg = QImage(arr.tobytes(), w, h, arr.strides[0], QImage.Format.Format_RGB888)
        return qimg.copy()
    return numpy_to_qimage(normalize_to_uint8(arr.squeeze()))


# ---------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------
class ImageCanvas(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.pix_item = QGraphicsPixmapItem()
        self.pix_item.setTransformationMode(Qt.TransformationMode.FastTransformation)
        self.scene().addItem(self.pix_item)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self._current_scale = 1.0
        self._min_scale = 1.0

    def set_pixmap(self, pm: QPixmap):
        self.scene().setSceneRect(QRectF(pm.rect()))
        self.pix_item.setPixmap(pm)
        self.reset_view()

    def reset_view(self):
        self.setTransform(QTransform())
        scene_rect = self.sceneRect()
        view_rect = self.viewport().rect()
        if scene_rect.width() > view_rect.width() or scene_rect.height() > view_rect.height():
            self.fitInView(scene_rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._current_scale = self.transform().m11()
        self._min_scale = self._current_scale

    def wheelEvent(self, event):
        if self.pix_item.pixmap().isNull():
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        new_scale = self._current_scale * factor
        if new_scale > 1.0:
            factor = 1.0 / self._current_scale
            self._current_scale = 1.0
        elif new_scale < self._min_scale:
            self.reset_view()
            return
        else:
            self._current_scale = new_scale
        self.scale(factor, factor)


# ---------------------------------------------------------------------
# Main Viewer
# ---------------------------------------------------------------------
class DICOMViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DICOM Viewer")
        self.resize(1200,800)

        # Multi-series storage
        self.series_data: Dict[str,Dict] = {}
        self.current_series: Optional[str] = None
        self.current_index: int = 0
        self.view_axis: str = "axial"
        self.volume: Optional[np.ndarray] = None

        # Tabs: Data and View
        self.series_list = QListWidget()
        self.series_list.currentItemChanged.connect(self.change_series)
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)
        data_layout.addWidget(self.series_list)

        self.canvas = ImageCanvas(self)
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.valueChanged.connect(self.on_slider_changed)
        view_tab = QWidget()
        view_layout = QVBoxLayout(view_tab)
        view_layout.addWidget(self.canvas)
        view_layout.addWidget(self.slice_slider)

        self.tabs = QTabWidget()
        self.tabs.addTab(data_tab, "Data")
        self.tabs.addTab(view_tab, "View")
        self.setCentralWidget(self.tabs)

        # Status bar
        self.info_label = QLabel("Ready")
        self.status = QStatusBar(self)
        self.status.addPermanentWidget(self.info_label,1)
        self.setStatusBar(self.status)

        self._build_toolbar()
        self._set_dark_theme()

        ok, det = jpeg2000_support_status()
        print("JPEG2000 support:", ok, det)
        if not ok:
            QMessageBox.information(self,"JPEG2000",
                "JPEG2000 decoding not available.\nInstall:\n"
                "  pip install pylibjpeg pylibjpeg-openjpeg\n"
                "  pip install python-gdcm\n\n"
                f"Detected: {det}")

    def _set_dark_theme(self):
        app = QApplication.instance()
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(53,53,53))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(25,25,25))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53,53,53))
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(53,53,53))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Highlight, QColor(142,45,197).lighter())
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        app.setPalette(palette)

    def _build_toolbar(self):
        tb = QToolBar("Main", self)
        tb.setIconSize(QSize(16,16))
        self.addToolBar(tb)

        act_open = QAction("Open Folder…", self)
        act_open.setShortcut(QKeySequence("Ctrl+O"))
        act_open.triggered.connect(self.open_folder_dialog)
        tb.addAction(act_open)

        tb.addSeparator()

        act_fit = QAction("Fit", self)
        act_fit.setShortcut("F")
        act_fit.triggered.connect(self.canvas.reset_view)
        tb.addAction(act_fit)

        act_prev = QAction("Prev", self)
        act_prev.setShortcuts([QKeySequence(Qt.Key.Key_Left), QKeySequence("PgUp")])
        act_prev.triggered.connect(self.prev_slice)
        tb.addAction(act_prev)

        act_next = QAction("Next", self)
        act_next.setShortcuts([QKeySequence(Qt.Key.Key_Right), QKeySequence("PgDown")])
        act_next.triggered.connect(self.next_slice)
        tb.addAction(act_next)

        tb.addSeparator()

        self.axis_combo = QComboBox()
        self.axis_combo.addItems(["Axial", "Coronal", "Sagittal"])
        self.axis_combo.currentTextChanged.connect(self.change_orientation)
        tb.addWidget(self.axis_combo)

    # -----------------------------------------------------------------
    # File loading
    # -----------------------------------------------------------------
    def open_folder_dialog(self):
        path = QFileDialog.getExistingDirectory(self,"Open Folder")
        if path: self.load_folder(path)

    def load_folder(self, folder: str):
        self.series_data.clear()
        self.series_list.clear()

        for root,_,files in os.walk(folder):
            for fn in files:
                path = os.path.join(root,fn)
                if not is_dicom_file(path): continue
                try:
                    ds = dcmread(path, stop_before_pixels=True, force=True)
                except Exception: continue
                key = series_key_from_ds(ds)
                self.series_data.setdefault(key,{"paths":[],"meta":ds})
                self.series_data[key]["paths"].append(path)

        # sort slices
        for key,sd in self.series_data.items():
            paths = sd["paths"]
            sort_list = []
            for p in paths:
                try: ds = dcmread(p, stop_before_pixels=True, force=True)
                except Exception: continue
                sort_list.append((sort_key_from_ds(ds), p))
            sort_list.sort(key=lambda x:x[0])
            sd["paths"] = [p for _,p in sort_list]

        # populate list
        for key,sd in self.series_data.items():
            ds = sd["meta"]
            desc = getattr(ds,"SeriesDescription","")
            pat = getattr(ds,"PatientID","?")
            mod = getattr(ds,"Modality","?")
            item = QListWidgetItem(f"{mod} | {desc} | Patient {pat} | {len(sd['paths'])} imgs")
            item.setData(Qt.ItemDataRole.UserRole,key)
            self.series_list.addItem(item)

        if self.series_list.count()>0:
            self.series_list.setCurrentRow(0)

    def change_series(self, current: QListWidgetItem):
        if not current: return
        key = current.data(Qt.ItemDataRole.UserRole)
        self.current_series = key
        self.current_index = 0
        paths = self.series_data[key]["paths"]
        vol = []
        for p in paths:
            try:
                ds = dcmread(p, force=True)
                vol.append(dicom_to_ndarray(ds))
            except Exception:
                continue
        self.volume = np.stack(vol, axis=0) if vol else None
        self._update_slider_range()
        self.display_current()

    # -----------------------------------------------------------------
    # Display
    # -----------------------------------------------------------------
    def display_current(self):
        if self.volume is None:
            return
        try:
            if self.view_axis == "axial":
                arr = self.volume[self.current_index]
            elif self.view_axis == "coronal":
                arr = self.volume[:, self.current_index, :]
            else:
                arr = self.volume[:, :, self.current_index]
            qimg = numpy_to_qimage(arr)
            self.canvas.set_pixmap(QPixmap.fromImage(qimg))
            total = self.volume.shape[0] if self.view_axis == "axial" else (
                self.volume.shape[1] if self.view_axis == "coronal" else self.volume.shape[2]
            )
            info = f"Slice {self.current_index+1}/{total} ({self.view_axis})"
            self.info_label.setText(info)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self,"Display error",str(e))

    def on_slider_changed(self,val:int):
        self.current_index = val
        self.display_current()

    def next_slice(self):
        if self.current_series:
            if self.current_index < self.slice_slider.maximum():
                self.current_index += 1
                self.slice_slider.setValue(self.current_index)

    def prev_slice(self):
        if self.current_series and self.current_index>0:
            self.current_index -= 1
            self.slice_slider.setValue(self.current_index)

    def change_orientation(self, text: str):
        self.view_axis = text.lower()
        self.current_index = 0
        self._update_slider_range()
        self.display_current()

    def _update_slider_range(self):
        if self.volume is None:
            self.slice_slider.setMaximum(0)
            return
        if self.view_axis == "axial":
            n = self.volume.shape[0]
        elif self.view_axis == "coronal":
            n = self.volume.shape[1]
        else:
            n = self.volume.shape[2]
        self.slice_slider.setMaximum(max(0, n - 1))


def main():
    app = QApplication(sys.argv)
    win = DICOMViewer()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
  
