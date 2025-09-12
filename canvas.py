"""Graphics view used to display slices."""

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QTransform, QPixmap, QPainter
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem


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

    def wheelEvent(self, event):  # noqa: N802 (Qt API)
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
