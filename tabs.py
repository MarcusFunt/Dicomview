from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QListWidget,
    QSlider,
    QPushButton,
    QTabWidget,
)


class ViewerTabs:
    """Tab widget containing Data, View and Prep panels."""

    def __init__(self, parent, canvas):
        self.series_list = QListWidget()
        self.series_list.currentItemChanged.connect(parent.change_series)
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)
        data_layout.addWidget(self.series_list)

        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.valueChanged.connect(parent.on_slider_changed)
        view_tab = QWidget()
        view_layout = QVBoxLayout(view_tab)
        view_layout.addWidget(canvas)
        view_layout.addWidget(self.slice_slider)

        self.normalize_button = QPushButton("Normalize Intensity")
        self.normalize_button.clicked.connect(parent.normalize_volume)
        self.nppy_button = QPushButton("Run Neural Pre-Processing")
        self.nppy_button.clicked.connect(parent.run_nppy)
        prep_tab = QWidget()
        prep_layout = QVBoxLayout(prep_tab)
        prep_layout.addWidget(self.normalize_button)
        prep_layout.addWidget(self.nppy_button)
        prep_layout.addStretch()

        tabs = QTabWidget()
        tabs.addTab(data_tab, "Data")
        tabs.addTab(view_tab, "View")
        tabs.addTab(prep_tab, "Prep")
        tabs.currentChanged.connect(parent.update_toolbar_visibility)
        parent.setCentralWidget(tabs)

        self.tabs = tabs
