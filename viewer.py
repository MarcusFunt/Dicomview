"""Modernized PyQt6 DICOM/Image Viewer.

- Multi-series browser (select series from folder).
- Supports JPEG2000 (lossless & irreversible) via pylibjpeg-openjpeg or GDCM.
- Dark theme, modern split layout, toolbar.
- Zoom (wheel), pan (drag), slice slider, drag-and-drop.
"""

import os
import sys
from typing import Dict, Optional

import numpy as np
import pydicom
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QPixmap, QKeySequence, QPalette, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QToolBar,
    QLabel,
    QSlider,
    QWidget,
    QVBoxLayout,
    QMessageBox,
    QStatusBar,
    QListWidget,
    QListWidgetItem,
    QComboBox,
    QTabWidget,
)

from canvas import ImageCanvas
from utils import (
    jpeg2000_support_status,
    is_dicom_file,
    series_key_from_ds,
    sort_key_from_ds,
    dicom_to_ndarray,
    numpy_to_qimage,
)


class DICOMViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DICOM Viewer")
        self.resize(1200, 800)

        # Multi-series storage
        self.series_data: Dict[str, Dict] = {}
        self.current_series: Optional[str] = None
        self.current_index: int = 0
        self.view_axis: str = "axial"
        self.volume: Optional[np.ndarray] = None
        self.series_is_3d: bool = False
        self.invert_colors: bool = False

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
        self.tabs.currentChanged.connect(self.update_toolbar_visibility)
        self.setCentralWidget(self.tabs)

        # Status bar
        self.info_label = QLabel("Ready")
        self.status = QStatusBar(self)
        self.status.addPermanentWidget(self.info_label, 1)
        self.setStatusBar(self.status)

        self._build_sidebar()
        self.update_toolbar_visibility()
        self._set_dark_theme()

        ok, det = jpeg2000_support_status()
        print("JPEG2000 support:", ok, det)
        if not ok:
            QMessageBox.information(
                self,
                "JPEG2000",
                "JPEG2000 decoding not available.\nInstall:\n"
                "  pip install pylibjpeg pylibjpeg-openjpeg\n"
                "  pip install python-gdcm\n\n"
                f"Detected: {det}",
            )

    def _set_dark_theme(self):
        app = QApplication.instance()
        app.setStyle("Fusion")
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Highlight, QColor(142, 45, 197).lighter())
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        app.setPalette(palette)

    def _build_sidebar(self):
        tb = QToolBar("Main", self)
        tb.setMovable(False)
        tb.setIconSize(QSize(24, 24))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, tb)

        self.act_open = QAction("Open", self)
        self.act_open.setShortcut(QKeySequence("Ctrl+O"))
        self.act_open.triggered.connect(self.open_folder_dialog)
        tb.addAction(self.act_open)
        self.sep_after_open = tb.addSeparator()

        self.act_prev = QAction("Prev", self)
        self.act_prev.setShortcuts([QKeySequence(Qt.Key.Key_Left), QKeySequence("PgUp")])
        self.act_prev.triggered.connect(self.prev_slice)
        tb.addAction(self.act_prev)

        self.act_next = QAction("Next", self)
        self.act_next.setShortcuts([QKeySequence(Qt.Key.Key_Right), QKeySequence("PgDown")])
        self.act_next.triggered.connect(self.next_slice)
        tb.addAction(self.act_next)

        self.sep_before_axis = tb.addSeparator()

        self.axis_combo = QComboBox()
        self.axis_combo.addItems(["Axial", "Coronal", "Sagittal"])
        self.axis_combo.currentTextChanged.connect(self.change_orientation)
        tb.addWidget(self.axis_combo)

        self.sep_before_zoom = tb.addSeparator()
        self.act_zoom_in = QAction("Zoom +", self)
        self.act_zoom_in.triggered.connect(self.canvas.zoom_in)
        tb.addAction(self.act_zoom_in)
        self.act_zoom_out = QAction("Zoom -", self)
        self.act_zoom_out.triggered.connect(self.canvas.zoom_out)
        tb.addAction(self.act_zoom_out)
        self.act_reset = QAction("Reset", self)
        self.act_reset.triggered.connect(self.canvas.reset_view)
        tb.addAction(self.act_reset)

        self.sep_before_config = tb.addSeparator()
        self.act_invert = QAction("Invert", self, checkable=True)
        self.act_invert.toggled.connect(self.toggle_invert)
        tb.addAction(self.act_invert)

    def update_toolbar_visibility(self, index: int = None):
        is_data = self.tabs.currentIndex() == 0
        self.act_open.setVisible(is_data)
        self.sep_after_open.setVisible(is_data)

        show_nav = not is_data
        self.act_prev.setVisible(show_nav)
        self.act_next.setVisible(show_nav)
        self.sep_before_axis.setVisible(show_nav and self.series_is_3d)
        self.axis_combo.setVisible(show_nav and self.series_is_3d)

        self.sep_before_zoom.setVisible(show_nav)
        self.act_zoom_in.setVisible(show_nav)
        self.act_zoom_out.setVisible(show_nav)
        self.act_reset.setVisible(show_nav)
        self.sep_before_config.setVisible(show_nav)
        self.act_invert.setVisible(show_nav)

    # -----------------------------------------------------------------
    # File loading
    # -----------------------------------------------------------------
    def open_folder_dialog(self):
        path = QFileDialog.getExistingDirectory(self, "Open Folder")
        if path:
            self.load_folder(path)

    def load_folder(self, folder: str):
        self.series_data.clear()
        self.series_list.clear()

        for root, _, files in os.walk(folder):
            for fn in files:
                path = os.path.join(root, fn)
                if not is_dicom_file(path):
                    continue
                try:
                    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
                except Exception:
                    continue
                key = series_key_from_ds(ds)
                self.series_data.setdefault(key, {"paths": [], "meta": ds})
                self.series_data[key]["paths"].append(path)

        # sort slices
        for key, sd in self.series_data.items():
            paths = sd["paths"]
            sort_list = []
            for p in paths:
                try:
                    ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
                except Exception:
                    continue
                sort_list.append((sort_key_from_ds(ds), p))
            sort_list.sort(key=lambda x: x[0])
            sd["paths"] = [p for _, p in sort_list]

        # populate list
        for key, sd in self.series_data.items():
            ds = sd["meta"]
            desc = getattr(ds, "SeriesDescription", "")
            pat = getattr(ds, "PatientID", "?")
            mod = getattr(ds, "Modality", "?")
            item = QListWidgetItem(
                f"{mod} | {desc} | Patient {pat} | {len(sd['paths'])} imgs"
            )
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.series_list.addItem(item)

        if self.series_list.count() > 0:
            self.series_list.setCurrentRow(0)

    def change_series(self, current: QListWidgetItem):
        if not current:
            return
        key = current.data(Qt.ItemDataRole.UserRole)
        self.current_series = key
        self.current_index = 0

        self.series_is_3d = "3d" in current.text().lower()
        self.axis_combo.setCurrentIndex(0)
        self.view_axis = "axial"
        self.update_toolbar_visibility()

        paths = self.series_data[key]["paths"]
        vol = []
        for p in paths:
            try:
                ds = pydicom.dcmread(p, force=True)
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
            if self.invert_colors:
                arr = 255 - arr
            qimg = numpy_to_qimage(arr)
            self.canvas.set_pixmap(QPixmap.fromImage(qimg))
            total = (
                self.volume.shape[0]
                if self.view_axis == "axial"
                else (
                    self.volume.shape[1]
                    if self.view_axis == "coronal"
                    else self.volume.shape[2]
                )
            )
            info = f"Slice {self.current_index+1}/{total} ({self.view_axis})"
            self.info_label.setText(info)
        except Exception as e:  # pragma: no cover - GUI feedback
            import traceback

            traceback.print_exc()
            QMessageBox.critical(self, "Display error", str(e))

    def on_slider_changed(self, val: int):
        self.current_index = val
        self.display_current()

    def toggle_invert(self, checked: bool):
        self.invert_colors = checked
        self.display_current()

    def next_slice(self):
        if self.current_series:
            if self.current_index < self.slice_slider.maximum():
                self.current_index += 1
                self.slice_slider.setValue(self.current_index)

    def prev_slice(self):
        if self.current_series and self.current_index > 0:
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
