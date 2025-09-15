"""Graphics view used to display slices."""

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QTransform, QPixmap, QPainter
from PyQt6.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
)


class ImageCanvas(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.pix_item = QGraphicsPixmapItem()
        self.pix_item.setTransformationMode(
            Qt.TransformationMode.FastTransformation
        )
        self.scene().addItem(self.pix_item)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self._zoom = 1.0

    def set_pixmap(self, pm: QPixmap, reset: bool = True):
        self.scene().setSceneRect(QRectF(pm.rect()))
        self.pix_item.setPixmap(pm)
        if reset:
            self.reset_view()

    def reset_view(self):
        self._zoom = 1.0
        self.setTransform(QTransform())
        scene_rect = self.sceneRect()
        view_rect = self.viewport().rect()
        if scene_rect.width() > view_rect.width() or scene_rect.height() > view_rect.height():
            self.fitInView(scene_rect, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self._apply_zoom(factor)

    def zoom_in(self):
        self._apply_zoom(1.25)

    def zoom_out(self):
        self._apply_zoom(0.8)

    def _apply_zoom(self, factor: float):
        self.scale(factor, factor)
        self._zoom *= factor
