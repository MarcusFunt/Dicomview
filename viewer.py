"""Modernized PyQt6 DICOM/Image Viewer."""

import os
import sys
from typing import Dict, Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QPalette, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QLabel,
    QMessageBox,
    QStatusBar,
    QListWidgetItem,
)

from canvas import ImageCanvas
from utils import (
    jpeg2000_support_status,
    numpy_to_qimage,
)
from sidebar import Sidebar
from tabs import ViewerTabs
import files as fileio
import processing


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

        # Canvas and UI components
        self.canvas = ImageCanvas(self)
        self.tabs = ViewerTabs(self, self.canvas)
        self.sidebar = Sidebar(self, self.canvas)

        # References for convenience
        self.series_list = self.tabs.series_list
        self.slice_slider = self.tabs.slice_slider
        self.normalize_button = self.tabs.normalize_button
        self.nppy_button = self.tabs.nppy_button
        self.tab_widget = self.tabs.tabs

        # Status bar
        self.info_label = QLabel("Ready")
        self.status = QStatusBar(self)
        self.status.addPermanentWidget(self.info_label, 1)
        self.setStatusBar(self.status)

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

    def update_toolbar_visibility(self, index: int = None):
        current = self.tab_widget.currentIndex() if index is None else index
        is_data = current == 0
        is_view = current == 1
        self.sidebar.update_visibility(is_data, is_view, self.series_is_3d)

    # -----------------------------------------------------------------
    # File loading
    # -----------------------------------------------------------------
    def open_folder_dialog(self):
        path = QFileDialog.getExistingDirectory(self, "Open Folder")
        if path:
            self.load_folder(path)

    def load_folder(self, folder: str):
        self.series_data = fileio.discover_series(folder)
        self.series_list.clear()

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
        ds = self.series_data[key]["meta"]
        desc = getattr(ds, "SeriesDescription", "")
        self.series_is_3d = "3d" in desc.lower()
        self.sidebar.axis_combo.setCurrentIndex(0)
        self.view_axis = "axial"
        self.update_toolbar_visibility()

        paths = self.series_data[key]["paths"]
        self.volume = fileio.load_volume(paths)
        self._update_slider_range()
        self.display_current(reset_view=True)

    # -----------------------------------------------------------------
    # Display
    # -----------------------------------------------------------------
    def display_current(self, reset_view: bool = False):
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
            self.canvas.set_pixmap(QPixmap.fromImage(qimg), reset=reset_view)
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

    # -----------------------------------------------------------------
    # Pre-processing
    # -----------------------------------------------------------------
    def normalize_volume(self):
        self.volume = processing.normalize_volume(self.volume)
        self.display_current()

    def run_nppy(self):
        if not self.current_series:
            return
        paths = self.series_data[self.current_series]["paths"]
        input_folder = os.path.commonpath(paths)
        output_folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not output_folder:
            return
        try:
            processing.run_nppy(input_folder, output_folder)
            QMessageBox.information(
                self, "Neural Pre-Processing", f"Results saved to: {output_folder}"
            )
        except Exception as e:  # pragma: no cover - GUI feedback
            QMessageBox.critical(self, "Neural Pre-Processing", str(e))

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
        self.display_current(reset_view=True)

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