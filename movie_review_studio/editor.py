"""Scrollable, frame-indexed timeline and virtual scene grid."""
from bisect import bisect_right
from functools import lru_cache
import math

from PySide6.QtCore import QAbstractListModel, QModelIndex, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from .analysis import thumbnail_path
from .render import audio_spans
from .smart import PEAKS_PER_SECOND


BLUE = '#3164a3'
ORANGE = '#bd652e'
GREEN = '#387b58'
PURPLE = '#734f99'
LABEL_WIDTH = 5
RULER_HEIGHT = 25
TRACK_HEIGHTS = (57, 29, 29, 29)


@lru_cache(maxsize=250)
def thumbnail(cache, key, start):
    return QPixmap(str(thumbnail_path(cache, key, start)))


class SceneModel(QAbstractListModel):
    def __init__(self, studio):
        super().__init__(studio)
        self.studio = studio
        self.rows = []

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row, clip, label = self.rows[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return label
        if role == Qt.ItemDataRole.DecorationRole:
            return QIcon(thumbnail(self.studio.cache, self.studio.source_key, clip.start))
        if role == Qt.ItemDataRole.UserRole:
            return row
        return None

    def update_rows(self, rows):
        self.beginResetModel()
        self.rows = rows
        self.endResetModel()


class TrackLabels(QWidget):
    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self.setFixedWidth(106)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#202b3a'))
        painter.setPen(QColor('#dbe5f3'))
        for text, y in (('VIDEO', 53), ('ORIGINAL', 101), ('NARRATION', 124), ('MUSIC', 159)):
            painter.drawText(10, y, text)
        if self.studio.voice is None:
            painter.setPen(QColor('#a9b8cd'))
            painter.drawText(10, 137, 'Placeholder')


class Timeline(QWidget):
    seek = Signal(float)
    selected = Signal(int)
    moved = Signal(int, int)
    preview = Signal()

    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self.total = 0
        self.frame = None
        self.starts = []
        self.regions = []
        self.selection = -1
        self.zoom = 1.0
        self.press = None
        self.drag_row = None
        self.drop_row = None
        self.setMinimumHeight(RULER_HEIGHT + sum(TRACK_HEIGHTS) + 4)
        self.setMouseTracking(True)
        self.setAccessibleName('Montage timeline')

    def set_clips(self, clips):
        self.total = sum(clip.frames for clip in clips)
        self.starts = [entry.position for entry in self.studio.entries]
        self.set_frame(None)
        self.update_width()

    def update_width(self):
        if hasattr(self.studio, 'timeline_scroll'):
            width = self.studio.timeline_scroll.viewport().width()
            self.setMinimumWidth(max(0, int(width * self.zoom)))
            self.resize(max(width, self.minimumWidth()), self.height())
        self.update()

    def set_frame(self, frame):
        self.frame = frame
        self.update()

    def x_at(self, frame):
        return LABEL_WIDTH + frame / max(1, self.total) * (self.width() - LABEL_WIDTH - 8)

    def frame_at(self, x):
        return min(self.total, max(0, (x - LABEL_WIDTH) / max(1, self.width() - LABEL_WIDTH - 8) * self.total))

    def row_at(self, x):
        return min(len(self.starts) - 1, bisect_right(self.starts, self.frame_at(x)) - 1)

    def update_regions(self):
        studio = self.studio
        self.regions = [[], [], []]
        if not studio.movie:
            return
        fps = float(studio.movie.fps)
        cursor = 0
        for clip in studio.clips:
            if clip.movie_audio:
                self.regions[0].append((cursor, cursor + clip.frames,
                                        max(0, clip.start / fps + studio.analysis.offset), False))
            cursor += clip.frames
        narrated = 0
        for original, start, end in audio_spans(studio.clips):
            if not original:
                available = round(studio.waves['voice'].duration * fps) - narrated
                if available > 0:
                    self.regions[1].append((start, min(end, start + available), narrated / fps, False))
                narrated += end - start
        if studio.music and self.total:
            self.regions[2].append((0, self.total, 0, True))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(event.rect(), QColor('#111720'))
        for y, height in zip((25, 82, 111, 140), TRACK_HEIGHTS):
            painter.fillRect(0, y, self.width(), height - 1, QColor('#192331'))
        if self.total:
            self.paint_ruler(painter, event.rect())
            self.paint_clips(painter, event.rect())
            for index, key in enumerate(('original', 'voice', 'music')):
                if self.regions:
                    self.paint_waveforms(painter, event.rect(), self.regions[index],
                                         self.studio.waves[key].peaks, 82 + index * 29,
                                         (ORANGE, BLUE, GREEN)[index])
            if self.frame is not None:
                x = round(self.x_at(self.frame))
                painter.setPen(QPen(QColor('#f8e499'), 2))
                painter.drawLine(x, 9, x, self.height())
        else:
            painter.setPen(QColor('#a9b8cd'))
            painter.drawText(LABEL_WIDTH + 14, 57, 'Generate a review to populate the timeline.')

    def paint_ruler(self, painter, rect):
        fps = float(self.studio.movie.fps)
        per_second = (self.x_at(self.total) - self.x_at(0)) / (self.total / fps)
        interval = next((n for n in (1, 2, 5, 10, 15, 30, 60, 120, 180, 300) if n * per_second >= 65), 600)
        first = math.floor(self.frame_at(rect.left()) / fps / interval) * interval
        for second in range(first, int(self.frame_at(rect.right()) / fps) + 1, interval):
            x = round(self.x_at(second * fps))
            painter.setPen(QColor('#8191a8'))
            painter.drawLine(x, 17, x, self.height())
            painter.setPen(QColor('#c9d6e8'))
            painter.drawText(x + 4, 15, f'{second // 60:02}:{second % 60:02}')

    def paint_clips(self, painter, rect):
        start = max(0, self.row_at(rect.left()))
        for row in range(start, len(self.starts)):
            entry = self.studio.entries[row]
            clip = entry.segment
            left = self.x_at(entry.position)
            right = self.x_at(entry.position + clip.frames)
            if left > rect.right():
                break
            box = QRectF(left, 28, max(1, right - left - 1), 51)
            painter.fillRect(box, QColor(PURPLE if clip.role else ORANGE if clip.movie_audio else BLUE))
            if box.width() > 42:
                image = thumbnail(self.studio.cache, self.studio.source_key, clip.start)
                if not image.isNull():
                    target = QRectF(left + 2, 30, min(84, box.width() - 4), 47)
                    painter.drawPixmap(target, image, QRectF(image.rect()))
                painter.setPen(QColor('#ffffff'))
                painter.drawText(QRectF(left + 4, 30, box.width() - 5, 46),
                                 Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom, clip.role or str(row + 1))
            if row in (self.selection, self.studio.active_row, self.drop_row):
                painter.setPen(QPen(QColor('#ffdc80' if row == self.studio.active_row else '#ffffff'), 2))
                painter.drawRect(box.adjusted(1, 1, -1, -1))

    def paint_waveforms(self, painter, rect, regions, peaks, y, color):
        fps = float(self.studio.movie.fps)
        for start, end, offset, repeat in regions:
            left, right = round(self.x_at(start)), round(self.x_at(end))
            if right < rect.left() or left > rect.right():
                continue
            painter.fillRect(left, y + 2, max(1, right - left), 25, QColor(color))
            if not peaks:
                continue
            painter.setPen(QColor('#d8f1eb'))
            for x in range(max(left, rect.left()), min(right, rect.right() + 1), 2):
                index = int((offset + (self.frame_at(x) - start) / fps) * PEAKS_PER_SECOND)
                if repeat:
                    index %= len(peaks)
                if 0 <= index < len(peaks):
                    height = max(1, round(peaks[index] * 22))
                    painter.drawLine(x, y + 14 - height // 2, x, y + 14 + height // 2)

    def mousePressEvent(self, event):
        if self.total and event.button() == Qt.MouseButton.LeftButton and event.position().x() >= LABEL_WIDTH:
            self.press = event.position()
            if 25 <= event.position().y() < 82:
                self.drag_row = self.row_at(event.position().x())
                self.selected.emit(self.drag_row)
            else:
                self.seek.emit(self.frame_at(event.position().x()) / self.total)

    def mouseMoveEvent(self, event):
        if self.press is None:
            return
        if self.drag_row is None:
            self.seek.emit(self.frame_at(event.position().x()) / self.total)
        elif (event.position() - self.press).manhattanLength() > 6:
            self.drop_row = self.row_at(event.position().x())
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drag_row is not None:
            if self.drop_row is not None and self.drop_row != self.drag_row:
                self.moved.emit(self.drag_row, self.drop_row)
            elif self.drop_row is None:
                self.seek.emit(self.starts[self.drag_row] / self.total)
        scrubbed = self.press is not None and self.drag_row is None
        self.press = self.drag_row = self.drop_row = None
        if scrubbed:
            self.seek.emit(self.frame_at(event.position().x()) / self.total)
        self.update()

    def mouseDoubleClickEvent(self, event):
        if self.total and 25 <= event.position().y() < 82 and event.position().x() >= LABEL_WIDTH:
            self.selected.emit(self.row_at(event.position().x()))
            self.preview.emit()
