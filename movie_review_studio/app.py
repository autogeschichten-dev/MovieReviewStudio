"""Qt desktop workflow for importing, editing, previewing and exporting one review."""
from pathlib import Path
import json
import logging
import math
import sys

from PySide6.QtCore import QObject, QRunnable, QStandardPaths, QThreadPool, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFormLayout, QGridLayout, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QSlider, QSpinBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from .analysis import analyze, atomic_json, fingerprint_highlight
from .media import identity, inspect, invoke
from .model import adjacent_conflict, choose, overlaps, replacement, validate
from .render import render


class JobSignals(QObject):
    progress = Signal(str)
    result = Signal(object)
    error = Signal(str)


class Job(QRunnable):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.signals = JobSignals()

    def run(self):
        try:
            result = self.operation(self.signals.progress.emit)
        except Exception as error:
            logging.exception("Background operation failed")
            self.signals.error.emit(str(error))
        else:
            self.signals.result.emit(result)


def clock(seconds):
    milliseconds = max(0, round(seconds * 1000))
    minutes, remainder = divmod(milliseconds, 60000)
    whole, fraction = divmod(remainder, 1000)
    return f"{minutes // 60:02}:{minutes % 60:02}:{whole:02}.{fraction:03}"


def parse_time(value):
    parts = value.strip().split(":")
    if len(parts) > 3:
        raise ValueError("Use HH:MM:SS.sss, MM:SS.sss, or seconds.")
    numbers = [float(part) for part in parts]
    if any(not math.isfinite(part) or part < 0 for part in numbers):
        raise ValueError("Enter a valid positive source time.")
    return sum(part * 60 ** index for index, part in enumerate(reversed(numbers)))


class Timeline(QWidget):
    seek = Signal(float)

    def __init__(self):
        super().__init__()
        self.setFixedHeight(25)
        self.setToolTip("Montage timeline. Play the assembly to scrub. Gold segments use movie audio.")
        self.clips = []
        self.total = 0
        self.frame = None

    def set_clips(self, clips):
        self.clips = clips
        self.total = sum(clip.frames for clip in clips)
        self.set_frame(None)

    def set_frame(self, frame):
        self.frame = frame
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#202733"))
        if not self.total:
            return
        cursor = 0
        for clip in self.clips:
            end = cursor + clip.frames
            left = round(cursor / self.total * self.width())
            right = round(end / self.total * self.width())
            current = self.frame is not None and cursor <= self.frame < end
            color = ("#ffe399" if current else "#c99d49") if clip.highlight else (
                "#81ecd9" if current else "#416d78"
            )
            painter.fillRect(left, 3, max(1, right - left), 19, QColor(color))
            if right - left >= 3:
                painter.fillRect(right - 1, 3, 1, 19, QColor("#202733"))
            cursor = end
        if self.frame is not None:
            x = min(self.width() - 1, round(self.frame / self.total * self.width()))
            painter.fillRect(x, 0, 2, self.height(), QColor("#ffffff"))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.scrub(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.scrub(event)

    def scrub(self, event):
        if self.frame is not None and self.total:
            fraction = max(0.0, min(1.0, event.position().x() / max(1, self.width() - 1)))
            self.seek.emit(fraction)


class Studio(QMainWindow):
    def __init__(self, cache):
        super().__init__()
        self.cache = cache
        self.movie = self.analysis = self.source_key = None
        self.voice = self.music = None
        self.highlights = []
        self.clips = []
        self.playing_clips = []
        self.job = None
        self.source_playback = False
        self.dragging = False
        self.setWindowTitle("Movie Review Studio")
        self.resize(1240, 725)
        self.setMinimumSize(960, 640)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.import_tab(), "Import")
        self.tabs.addTab(self.cut_tab(), "Cut")
        self.tabs.addTab(self.final_tab(), "Final")

        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        video = QVideoWidget()
        video.setMinimumHeight(220)
        self.player.setVideoOutput(video)
        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(lambda duration: self.seek.setMaximum(duration))
        self.player.errorOccurred.connect(self.playback_error)
        self.seek = QSlider(Qt.Orientation.Horizontal)
        self.seek.setRange(0, 0)
        self.seek.sliderPressed.connect(lambda: setattr(self, "dragging", True))
        self.seek.sliderReleased.connect(self.seek_released)
        self.seek.actionTriggered.connect(self.seek_action)
        self.position = QLabel(clock(0))
        self.preview_label = QLabel("Preview")
        self.preview_label.setObjectName("previewTitle")
        preview = QVBoxLayout()
        preview.addWidget(self.preview_label)
        preview.addWidget(video, 1)
        preview.addWidget(self.seek)
        transport = QHBoxLayout()
        transport.addWidget(self.button("Play / Pause", self.toggle_playback))
        transport.addWidget(self.button("Source preview", self.preview_source))
        transport.addWidget(self.position, 1, Qt.AlignmentFlag.AlignRight)
        preview.addLayout(transport)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        self.progress.hide()
        layout = QVBoxLayout()
        workspace = QHBoxLayout()
        workspace.addWidget(self.tabs, 6)
        workspace.addLayout(preview, 5)
        workspace.setSpacing(18)
        layout.addLayout(workspace, 1)
        layout.addWidget(self.progress)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Import a local movie to begin.")
        self.refresh()

    def button(self, title, action):
        control = QPushButton(title)
        control.clicked.connect(lambda: self.perform(action))
        return control

    @staticmethod
    def panel(layout):
        panel = QWidget()
        panel.setLayout(layout)
        return panel

    def import_tab(self):
        form = QFormLayout()
        self.title = QLineEdit()
        self.title.setPlaceholderText("Movie title")
        self.year = QSpinBox()
        self.year.setRange(0, 9999)
        self.year.setSpecialValueText("Unspecified")
        form.addRow("Title", self.title)
        form.addRow("Release year", self.year)
        self.movie_label = QLabel("No movie imported")
        self.movie_label.setWordWrap(True)
        self.voice_label = QLabel("Optional")
        self.music_label = QLabel("Optional")
        assets = (
            ("Movie", self.movie_label, self.import_movie, None),
            ("Voice-over", self.voice_label, lambda: self.import_audio("voice"), lambda: self.clear_audio("voice")),
            ("Music", self.music_label, lambda: self.import_audio("music"), lambda: self.clear_audio("music")),
        )
        for name, label, action, clear in assets:
            row = QHBoxLayout()
            row.addWidget(self.button(f"Import {name.lower()}", action))
            if clear:
                row.addWidget(self.button("Clear", clear))
            row.addWidget(label, 1)
            form.addRow(name, row)
        self.target = QSpinBox()
        self.target.setRange(16, 18)
        self.target.setValue(17)
        self.target.setSuffix(" min")
        form.addRow("Target duration", self.target)
        self.analyze_button = self.button("Analyze entire movie", self.analyze_source)
        form.addRow(self.analyze_button)
        self.analysis_label = QLabel()
        self.analysis_label.setWordWrap(True)
        form.addRow("Shot analysis", self.analysis_label)
        form.addRow(self.button("Open saved edit…", self.open_project))
        return self.panel(form)

    def cut_tab(self):
        layout = QVBoxLayout()
        highlights = QGridLayout()
        self.highlight_list = QComboBox()
        self.start_time = QLineEdit()
        self.start_time.setPlaceholderText("HH:MM:SS.sss")
        self.length = QSpinBox()
        self.length.setRange(10, 15)
        self.length.setValue(12)
        self.length.setSuffix(" s")
        highlights.addWidget(QLabel("Original-audio highlights"), 0, 0, 1, 2)
        highlights.addWidget(self.highlight_list, 0, 2, 1, 2)
        highlights.addWidget(self.button("Use source playhead", self.capture_start), 1, 0)
        highlights.addWidget(self.start_time, 1, 1, 1, 2)
        highlights.addWidget(self.length, 1, 3)
        for column, (title, action) in enumerate((
            ("Add", self.add_highlight), ("Preview", self.preview_highlight),
            ("Replace", self.replace_highlight), ("Remove", self.remove_highlight),
        )):
            highlights.addWidget(self.button(title, action), 2, column)
        layout.addLayout(highlights)

        actions = QHBoxLayout()
        actions.addWidget(self.button("Generate montage", self.generate))
        actions.addWidget(self.button("Play assembly", lambda: self.preview(self.clips)))
        actions.addWidget(self.button("Save edit…", self.save_project))
        layout.addLayout(actions)
        self.timeline = Timeline()
        self.timeline.seek.connect(lambda fraction: self.player.setPosition(round(fraction * self.player.duration())))
        layout.addWidget(self.timeline)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["#", "Type", "Timeline", "Source in", "Length"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(25)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.cellDoubleClicked.connect(lambda *_: self.perform(self.play_clip))
        layout.addWidget(self.table, 1)
        edits = QHBoxLayout()
        edits.addWidget(self.button("Play selected", self.play_clip))
        edits.addWidget(self.button("Replace selected", self.replace_clip))
        edits.addWidget(self.button("Delete selected", self.delete_clip))
        layout.addLayout(edits)
        return self.panel(layout)

    def final_tab(self):
        layout = QVBoxLayout()
        self.final_summary = QLabel()
        self.final_summary.setWordWrap(True)
        layout.addWidget(self.final_summary)
        self.export_button = self.button("Validate and export MP4…", self.export)
        self.export_button.setObjectName("primary")
        layout.addWidget(self.export_button)
        layout.addWidget(self.button("Play assembly", lambda: self.preview(self.clips)))
        layout.addStretch(1)
        return self.panel(layout)

    def perform(self, action):
        if self.job is not None:
            return
        try:
            action()
        except Exception as error:
            self.error(str(error))

    def playback_error(self, _, message):
        logging.error("Preview: %s", message)
        self.statusBar().showMessage(f"Preview: {message}")

    def error(self, message):
        logging.error("%s", message)
        QMessageBox.warning(self, "Movie Review Studio", message)

    def background(self, message, operation, on_result):
        self.tabs.setEnabled(False)
        self.progress.show()
        self.statusBar().showMessage(message)
        self.job = Job(operation)
        self.job.signals.progress.connect(self.statusBar().showMessage)

        def finish():
            self.tabs.setEnabled(True)
            self.progress.hide()
            self.job = None
            self.statusBar().showMessage("Ready")

        def success(result):
            finish()
            self.perform(lambda: on_result(result))

        def failure(message):
            finish()
            self.error(message)

        self.job.signals.result.connect(success)
        self.job.signals.error.connect(failure)
        QThreadPool.globalInstance().start(self.job)

    def closeEvent(self, event):
        if self.job:
            self.statusBar().showMessage("Wait for the current operation to finish.")
            event.ignore()
        else:
            event.accept()

    def import_movie(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import movie", "", "Movies (*.mp4 *.mkv *.mov)")
        if not path:
            return
        if (self.clips or self.highlights) and QMessageBox.question(
            self, "Replace movie", "Importing a different movie clears the current cuts and highlights. Continue?"
        ) != QMessageBox.StandardButton.Yes:
            return
        def load(_):
            movie = inspect(Path(path))
            return movie, identity(movie.path)

        self.background("Reading movie metadata…", load, self.apply_movie)

    def apply_movie(self, result):
        self.movie, self.source_key = result
        self.analysis = None
        self.highlights = []
        self.clips = []
        self.refresh()
        self.preview_source()

    def import_audio(self, track):
        formats = "Audio (*.mp3 *.wav)" if track == "voice" else "Audio (*.mp3 *.wav *.aac *.m4a *.flac *.ogg)"
        path, _ = QFileDialog.getOpenFileName(self, f"Import {track}", "", formats)
        if path:
            def apply(media):
                setattr(self, track, media.path)
                self.invalidate_preview()
                self.refresh()

            self.background("Checking audio file…", lambda _: inspect(Path(path), video=False), apply)

    def clear_audio(self, track):
        setattr(self, track, None)
        self.invalidate_preview()
        self.refresh()

    def require_movie(self, analyzed=False):
        if not self.movie:
            raise ValueError("Import a movie first.")
        if identity(self.movie.path) != self.source_key:
            raise ValueError("The movie file changed. Import it again.")
        if analyzed and not self.analysis:
            raise ValueError("Analyze the entire movie in Import first.")

    def analyze_source(self):
        self.require_movie()
        self.background("Analyzing movie…", lambda progress: analyze(self.movie, self.cache, progress), self.set_analysis)

    def set_analysis(self, result):
        self.analysis = result
        self.refresh()
        self.tabs.setCurrentIndex(1)

    def capture_start(self):
        if not self.source_playback:
            raise ValueError("Open Source preview and position its playhead first.")
        self.start_time.setText(clock(self.player.position() / 1000 - (self.analysis.offset if self.analysis else 0)))

    def highlight_index(self):
        index = self.highlight_list.currentIndex()
        if index < 0:
            raise ValueError("Select an original-audio highlight.")
        return index

    def add_highlight(self):
        if len(self.highlights) >= 3:
            raise ValueError("A review can contain up to three highlights.")
        self.read_highlight(None)

    def replace_highlight(self):
        self.read_highlight(self.highlight_index())

    def read_highlight(self, index):
        self.require_movie(True)
        start = round(parse_time(self.start_time.text()) * self.movie.fps)
        end = start + round(self.length.value() * self.movie.fps)
        # Quantize inward at 15 seconds and outward at 10 seconds.
        end = start + max(math.ceil(10 * self.movie.fps), min(int(15 * self.movie.fps), end - start))
        self.background("Reading highlight frames…",
                        lambda _: fingerprint_highlight(self.movie, self.analysis, start, end),
                        lambda clip: self.apply_highlight(index, clip))

    def apply_highlight(self, index, clip):
        updated = self.highlights.copy()
        edits = self.clips.copy()
        if index is None:
            updated.append(clip)
            if edits:
                if any(overlaps(clip, item) for item in edits):
                    raise ValueError("This highlight overlaps a timeline clip. Choose another range or delete the clip first.")
                for slot in range(len(edits), -1, -1):
                    if slot and adjacent_conflict(edits[slot - 1], clip):
                        continue
                    if slot < len(edits) and adjacent_conflict(clip, edits[slot]):
                        continue
                    edits.insert(slot, clip)
                    break
                else:
                    raise ValueError("The range conflicts with the current timeline. Choose another range.")
        else:
            previous = updated[index]
            updated[index] = clip
            if previous in edits:
                edits[edits.index(previous)] = clip
        ordered = sorted(updated, key=lambda item: item.start)
        if any(left.end > right.start for left, right in zip(ordered, ordered[1:])):
            raise ValueError("Original-audio highlights cannot overlap.")
        if edits:
            self.check(edits)
        self.highlights, self.clips = updated, edits
        self.invalidate_preview()
        self.refresh()
        self.highlight_list.setCurrentIndex(len(updated) - 1 if index is None else index)

    def preview_highlight(self):
        self.preview([self.highlights[self.highlight_index()]])

    def remove_highlight(self):
        index = self.highlight_index()
        self.remove_edit(self.highlights[index])

    def generate(self):
        self.require_movie(True)
        if self.clips and QMessageBox.question(
            self, "Regenerate montage", "Replace the current timeline with a new selection?"
        ) != QMessageBox.StandardButton.Yes:
            return
        target = self.target.value() * 60
        self.background("Selecting footage…", lambda _: choose(self.analysis.candidates, self.highlights,
                        self.movie.fps, self.analysis.frames, target), self.set_clips)

    def set_clips(self, clips):
        self.clips = clips
        self.invalidate_preview()
        self.refresh()

    def check(self, clips, final=False):
        validate(clips, self.movie.fps, self.analysis.frames, final=final, has_audio=self.movie.has_audio)

    def selected_row(self):
        index = self.table.currentRow()
        if not 0 <= index < len(self.clips):
            raise ValueError("Select a timeline clip.")
        return index

    def play_clip(self):
        self.preview([self.clips[self.selected_row()]])

    def replace_clip(self):
        index = self.selected_row()
        if self.clips[index].highlight:
            self.read_highlight(self.highlights.index(self.clips[index]))
            return
        edits = self.clips.copy()
        edits[index] = replacement(edits, index, self.analysis.candidates)
        self.check(edits)
        self.set_clips(edits)
        self.table.selectRow(index)

    def delete_clip(self):
        self.remove_edit(self.clips[self.selected_row()])

    def remove_edit(self, clip):
        edits = [item for item in self.clips if item != clip]
        if edits:
            self.check(edits)
        self.highlights = [item for item in self.highlights if item != clip]
        self.set_clips(edits)

    def preview_source(self):
        self.require_movie()
        self.load_player(self.movie.path, "Source movie", source=True)

    def preview(self, clips):
        self.require_movie(True)
        self.check(clips)
        assembled = clips == self.clips and len(clips) > 1
        voice = self.voice if assembled else None
        music = self.music if assembled else None
        self.background("Preparing preview…", lambda progress: render(self.movie, self.analysis, clips, voice,
                        music, self.cache, progress), lambda path: self.load_player(
                        path, "Montage preview" if assembled else "Selected clip", clips=clips))

    def load_player(self, path, label, *, source=False, clips=()):
        self.player.stop()
        self.source_playback = source
        self.playing_clips = list(clips)
        self.timeline.set_frame(0 if len(self.playing_clips) > 1 and self.playing_clips == self.clips else None)
        self.preview_label.setText(label)
        url = QUrl.fromLocalFile(str(path))
        if self.player.source() == url:
            self.player.setPosition(0)
        else:
            self.player.setSource(url)
        self.player.play()

    def invalidate_preview(self):
        if not self.source_playback:
            self.player.stop()
            self.player.setSource(QUrl())
            self.playing_clips = []
            self.timeline.set_frame(None)
            self.preview_label.setText("Preview")

    def toggle_playback(self):
        if not self.player.source().isValid():
            self.preview_source()
        elif self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def seek_released(self):
        self.dragging = False
        self.player.setPosition(self.seek.sliderPosition())

    def seek_action(self, action):
        if not self.dragging:
            self.player.setPosition(self.seek.sliderPosition())

    def position_changed(self, position):
        if not self.dragging:
            self.seek.setValue(position)
        self.position.setText(clock(position / 1000))
        if len(self.playing_clips) > 1 and self.playing_clips == self.clips:
            frame = position * self.movie.fps / 1000
            self.timeline.set_frame(frame)
            cursor = 0
            for index, clip in enumerate(self.playing_clips):
                cursor += clip.frames
                if frame < cursor:
                    if self.table.currentRow() != index:
                        self.table.selectRow(index)
                    break

    def refresh(self):
        movie = self.movie
        self.movie_label.setText(f"{movie.path.name}\n{movie.width} × {movie.height} · {float(movie.fps):.3f} FPS · {clock(movie.duration)}"
                                 if movie else "No movie imported")
        self.voice_label.setText(self.voice.name if self.voice else "Optional")
        self.music_label.setText(self.music.name if self.music else "Optional")
        self.analysis_label.setText(f"{len(self.analysis.candidates):,} usable candidates"
                                    if self.analysis else "Waiting for analysis")
        self.analyze_button.setEnabled(bool(movie))
        index = self.highlight_list.currentIndex()
        self.highlight_list.clear()
        self.highlight_list.addItems([f"{number}. {clock(clip.start / movie.fps)}"
                                     for number, clip in enumerate(self.highlights, 1)])
        if self.highlights:
            self.highlight_list.setCurrentIndex(max(0, min(index, len(self.highlights) - 1)))
        if self.timeline.clips != self.clips:
            self.timeline.set_clips(self.clips)
        self.table.setRowCount(len(self.clips))
        cursor = 0
        for index, clip in enumerate(self.clips):
            data = [str(index + 1), "Original audio" if clip.highlight else "Regular", clock(cursor / movie.fps),
                    clock(clip.start / movie.fps), f"{float(clip.frames / movie.fps):.2f} s"]
            for column, text in enumerate(data):
                item = QTableWidgetItem(text)
                if clip.highlight:
                    item.setForeground(Qt.GlobalColor.yellow)
                self.table.setItem(index, column, item)
            cursor += clip.frames
        self.final_summary.setText(f"{len(self.clips)} clips · {clock(cursor / movie.fps if movie else 0)}\n"
                                   f"{sum(clip.highlight for clip in self.clips)} original-audio highlights in timeline\n"
                                   "H.264 video · AAC stereo · MP4")
        self.export_button.setEnabled(bool(self.clips))

    def save_project(self):
        self.require_movie(True)
        path, _ = QFileDialog.getSaveFileName(self, "Save selected clips", "Review.mrs.json", "Review edit (*.mrs.json)")
        if not path:
            return
        destination = Path(path)
        if destination.resolve() in {item.resolve() for item in (self.movie.path, self.voice, self.music) if item}:
            raise ValueError("Save the edit to a different file from the imported media.")
        atomic_json(destination, {
            "version": 1, "movie": str(self.movie.path), "source_key": self.source_key,
            "voice": str(self.voice) if self.voice else None, "music": str(self.music) if self.music else None,
            "title": self.title.text(), "year": self.year.value(), "target": self.target.value(),
            "highlights": [[c.start, c.end] for c in self.highlights],
            "clips": [[c.start, c.end, c.highlight] for c in self.clips],
        })
        self.statusBar().showMessage(f"Saved edit to {destination}")

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open saved edit", "", "Review edit (*.mrs.json)")
        if not path:
            return
        if (self.clips or self.highlights) and QMessageBox.question(
            self, "Open another edit", "Opening this file replaces the current edit. Continue?"
        ) != QMessageBox.StandardButton.Yes:
            return

        def load(progress):
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if payload["version"] != 1:
                raise ValueError("Unsupported edit file version.")
            movie = inspect(Path(payload["movie"]))
            if identity(movie.path) != payload["source_key"]:
                raise ValueError("The saved edit references a different or changed movie file.")
            analysis = analyze(movie, self.cache, progress)
            highlights = [fingerprint_highlight(movie, analysis, start, end) for start, end in payload["highlights"]]
            indexed = {(c.start, c.end, c.highlight): c for c in analysis.candidates + highlights}
            clips = [indexed[tuple(item)] for item in payload["clips"]]
            if clips:
                validate(clips, movie.fps, analysis.frames, has_audio=movie.has_audio)
            ordered = sorted(highlights, key=lambda clip: clip.start)
            if len(ordered) > 3 or any(a.end > b.start for a, b in zip(ordered, ordered[1:])):
                raise ValueError("The saved highlights overlap or exceed the limit of three.")
            tracks = [Path(payload[name]) if payload[name] else None for name in ("voice", "music")]
            for track in tracks:
                if track:
                    inspect(track, video=False)
            return payload, movie, analysis, highlights, clips, tracks

        def apply(result):
            payload, self.movie, self.analysis, self.highlights, self.clips, tracks = result
            self.voice, self.music = tracks
            self.source_key = payload["source_key"]
            self.title.setText(payload["title"])
            self.year.setValue(payload["year"])
            self.target.setValue(payload["target"])
            self.source_playback = False
            self.invalidate_preview()
            self.refresh()
            self.tabs.setCurrentIndex(1)

        self.background("Opening saved edit…", load, apply)

    def export(self):
        self.require_movie(True)
        self.check(self.clips, final=True)
        path, _ = QFileDialog.getSaveFileName(self, "Export review", "Movie Review.mp4", "MP4 video (*.mp4)")
        if not path:
            return
        destination = Path(path)
        if destination.suffix.lower() != ".mp4":
            raise ValueError("Choose an .mp4 export filename.")
        self.player.stop()
        self.background("Exporting review…", lambda progress: render(
            self.movie, self.analysis, self.clips, self.voice, self.music, self.cache, progress,
            destination=destination), lambda output: QMessageBox.information(self, "Export complete", str(output)))


STYLE = """
QMainWindow, QWidget { background: #13171d; color: #f1f4f8; font: 13px 'Segoe UI'; }
QLabel { background: transparent; }
QLabel#previewTitle { font-size: 17px; font-weight: 700; }
QTabWidget::pane { border: 1px solid #394554; }
QTabBar::tab { padding: 9px 24px; background: #202733; }
QTabBar::tab:selected { background: #303e50; border-bottom: 3px solid #65dcca; }
QPushButton, QLineEdit, QSpinBox, QComboBox {
    background: #252e3a; border: 1px solid #4a596c; border-radius: 4px; padding: 6px;
}
QPushButton:hover { background: #3c4d61; }
QPushButton:focus, QLineEdit:focus, QSpinBox:focus, QComboBox:focus { border: 1px solid #65dcca; }
QPushButton:disabled { color: #a1aab6; background: #1d2530; }
QPushButton#primary { background: #65dcca; color: #10231f; font-weight: bold; }
QTableWidget { background: #1b222c; gridline-color: #323c49; selection-background-color: #3b526c; }
QHeaderView::section { background: #2b3543; color: white; padding: 5px; border: 0; }
QVideoWidget { background: black; }
QProgressBar::chunk { background: #65dcca; }
"""


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Movie Review Studio")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    data = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation))
    try:
        data.mkdir(exist_ok=True, parents=True)
        cache = data / "cache"
        cache.mkdir(exist_ok=True)
        logging.basicConfig(filename=data / "studio.log", encoding="utf-8", level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(message)s")
        encoders = invoke(["ffmpeg", "-hide_banner", "-encoders"])
        invoke(["ffprobe", "-version"])
        if not {"libx264", "aac"} <= {line.split()[1] for line in encoders.splitlines() if len(line.split()) >= 2}:
            raise RuntimeError("The FFmpeg build must include libx264 and AAC encoders.")
        window = Studio(cache)
        window.show()
        return app.exec()
    except Exception as error:
        logging.exception("Startup failed")
        QMessageBox.critical(None, "Movie Review Studio", f"{error}\nInstall FFmpeg and ensure ffmpeg and ffprobe are on PATH.")
        return 1
