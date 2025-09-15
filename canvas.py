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
