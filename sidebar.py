from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QToolBar, QComboBox


class Sidebar:
    """Toolbar with navigation and view controls."""

    def __init__(self, parent, canvas):
        tb = QToolBar("Main", parent)
        tb.setMovable(False)
        tb.setIconSize(QSize(24, 24))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        parent.addToolBar(Qt.ToolBarArea.RightToolBarArea, tb)

        self.act_open = QAction("Open", parent)
        self.act_open.setShortcut(QKeySequence("Ctrl+O"))
        self.act_open.triggered.connect(parent.open_folder_dialog)
        tb.addAction(self.act_open)
        self.sep_after_open = tb.addSeparator()

        self.act_prev = QAction("Prev", parent)
        self.act_prev.setShortcuts([QKeySequence(Qt.Key.Key_Left), QKeySequence("PgUp")])
        self.act_prev.triggered.connect(parent.prev_slice)
        tb.addAction(self.act_prev)

        self.act_next = QAction("Next", parent)
        self.act_next.setShortcuts([QKeySequence(Qt.Key.Key_Right), QKeySequence("PgDown")])
        self.act_next.triggered.connect(parent.next_slice)
        tb.addAction(self.act_next)

        self.act_zoom_in = QAction("Zoom +", parent)
        self.act_zoom_in.setShortcut(QKeySequence("+"))
        self.act_zoom_in.triggered.connect(canvas.zoom_in)
        tb.addAction(self.act_zoom_in)

        self.act_zoom_out = QAction("Zoom -", parent)
        self.act_zoom_out.setShortcut(QKeySequence("-"))
        self.act_zoom_out.triggered.connect(canvas.zoom_out)
        tb.addAction(self.act_zoom_out)

        self.sep_before_axis = tb.addSeparator()

        self.axis_combo = QComboBox()
        self.axis_combo.addItems(["Axial", "Coronal", "Sagittal"])
        self.axis_combo.currentTextChanged.connect(parent.change_orientation)
        tb.addWidget(self.axis_combo)

        self.toolbar = tb

    def update_visibility(self, is_data: bool, is_view: bool, show_axis: bool):
        """Show or hide actions based on current tab and data."""
        self.act_open.setVisible(is_data)
        self.sep_after_open.setVisible(is_data)

        self.act_prev.setVisible(is_view)
        self.act_next.setVisible(is_view)
        self.act_zoom_in.setVisible(is_view)
        self.act_zoom_out.setVisible(is_view)
        self.sep_before_axis.setVisible(is_view and show_axis)
        self.axis_combo.setVisible(is_view and show_axis)
