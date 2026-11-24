"""Qt desktop workflow for importing, editing, previewing and exporting one review."""
from dataclasses import replace
from pathlib import Path
from threading import Event
import os
import json
import logging
import math
import sys

from PySide6.QtCore import QObject, QRunnable, QSize, QStandardPaths, QThreadPool, Qt, QUrl, Signal
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut, QValidator
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFormLayout, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QListView, QMainWindow, QMessageBox, QProgressBar, QScrollArea,
    QPushButton, QSlider, QSpinBox, QSplitter, QStackedWidget, QTabWidget,
    QVBoxLayout, QWidget,
)

from .analysis import analyze, atomic_json, fingerprint_highlight
from .media import Cancelled, cancellation, identity, inspect, invoke
from .editor import SceneModel, Timeline, TrackLabels, thumbnail
from .model import adjacent_conflict, overlaps, replacement, timeline_entries, validate
from .render import Mix, render, render_selection
from .smart import AudioEvidence, LocalWhisper, assemble, audio_evidence, suggest_highlights


class JobSignals(QObject):
    progress = Signal(str)
    result = Signal(object)
    error = Signal(str)


class Job(QRunnable):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.signals = JobSignals()
        self.cancel = Event()

    def report(self, message):
        if self.cancel.is_set():
            raise Cancelled()
        self.signals.progress.emit(message)

    def run(self):
        token = cancellation.set(self.cancel)
        try:
            result = self.operation(self.report)
            if self.cancel.is_set():
                raise Cancelled()
        except Cancelled:
            self.signals.error.emit("")
        except Exception as error:
            logging.exception("Background operation failed")
            self.signals.error.emit(str(error))
        else:
            self.signals.result.emit(result)
        finally:
            cancellation.reset(token)


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


class TargetDuration(QSpinBox):
    def textFromValue(self, value):
        return f'{value // 60}:{value % 60:02}'

    def valueFromText(self, text):
        return round(parse_time(text))

    def validate(self, text, position):
        try:
            valid = self.minimum() <= self.valueFromText(text) <= self.maximum()
        except ValueError:
            valid = False
        state = QValidator.State.Acceptable if valid else QValidator.State.Intermediate
        return state, text, position


class Studio(QMainWindow):
    def __init__(self, cache):
        super().__init__()
        self.cache = cache
        self.movie = self.analysis = self.source_key = None
        self.voice = self.music = None
        self.highlights = []
        self.clips = []
        self.montage_playback = False
        self.entries = []
        self.active_row = None
        self.selected_clip_row = -1
        self.play_offset = 0
        self.play_entry = None
        self.pending_play = False
        self.browser_source = None
        self.mix = Mix()
        self.waves = dict.fromkeys(('original', 'voice', 'music'), AudioEvidence())
        self.undo_stack = []
        self.redo_stack = []
        self.pending_position = 0
        self.browser_preview_token = None
        self.job = None
        self.source_playback = False
        self.dragging = False
        self.setWindowTitle("Movie Review Studio")
        self.resize(1280, 740)
        self.setMinimumSize(960, 640)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.import_tab(), "Import")
        self.smart_page = QVBoxLayout()
        self.tabs.addTab(self.panel(self.smart_page), "Smart Editor")
        self.tabs.addTab(self.final_tab(), "Export")

        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.video = QVideoWidget()
        self.video.setMinimumHeight(136)
        self.player.setVideoOutput(self.video)
        self.video.videoSink().videoFrameChanged.connect(self.show_video_frame)
        self.video_stack = QStackedWidget()
        self.video_stack.addWidget(self.video)
        self.still = QLabel()
        self.still.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.still.setStyleSheet("background: #0b1017;")
        self.video_stack.addWidget(self.still)
        self.pause_on_frame = False
        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(lambda duration: self.seek.setMaximum(duration))
        self.player.mediaStatusChanged.connect(self.start_pending_preview)
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
        preview.addWidget(self.video_stack, 1)
        preview.addWidget(self.seek)
        transport = QHBoxLayout()
        transport.addWidget(self.button("Previous", lambda: self.step_clip(-1)))
        transport.addWidget(self.button("Play / Pause", self.toggle_playback))
        transport.addWidget(self.button("Next", lambda: self.step_clip(1)))
        transport.addWidget(self.button("Play review", lambda: self.preview(self.clips, assembly=True)))
        transport.addWidget(self.button("Source preview", self.preview_source))
        transport.addWidget(self.position, 1, Qt.AlignmentFlag.AlignRight)
        preview.addLayout(transport)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        self.progress.hide()
        self.build_smart_editor(preview)
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 10, 12, 7)
        layout.addWidget(self.tabs, 1)
        work_status = QHBoxLayout()
        work_status.addWidget(self.progress, 1)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_operation)
        self.cancel_button.hide()
        work_status.addWidget(self.cancel_button)
        layout.addLayout(work_status)
        self.setCentralWidget(self.panel(layout))
        for shortcut, action in ((QKeySequence.StandardKey.Undo, self.undo), (QKeySequence.StandardKey.Redo, self.redo)):
            QShortcut(QKeySequence(shortcut), self, activated=lambda fn=action: self.perform(fn))
        self.statusBar().showMessage("Import a local movie to begin.")
        self.refresh()
        self.title.textChanged.connect(lambda _: self.refresh_title())

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
        self.analyze_button = self.button("Auto Generate Movie Review", self.generate)
        self.analyze_button.setObjectName("primary")
        self.import_target = TargetDuration()
        self.import_target.setRange(1020, 1080)
        self.import_target.setValue(1050)
        self.import_target.setSingleStep(5)
        form.addRow("Target duration", self.import_target)
        form.addRow(self.analyze_button)
        self.analysis_label = QLabel()
        self.analysis_label.setWordWrap(True)
        form.addRow("Analysis", self.analysis_label)
        form.addRow(self.button("Open saved project…", self.open_project))
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

        return self.panel(layout)

    def build_smart_editor(self, preview):
        header = QHBoxLayout()
        self.editor_title = QLabel()
        self.editor_title.setObjectName("previewTitle")
        header.addWidget(self.editor_title, 1)
        header.addWidget(QLabel("Target"))
        self.target = TargetDuration()
        self.target.setRange(1020, 1080)
        self.target.setValue(1050)
        self.target.setSingleStep(5)
        self.target.setFixedWidth(93)
        self.target.valueChanged.connect(self.import_target.setValue)
        self.import_target.valueChanged.connect(self.target.setValue)
        header.addWidget(self.target)
        generate = self.button("Auto Generate Movie Review", self.generate)
        generate.setObjectName("primary")
        header.addWidget(generate)
        header.addWidget(self.button("Save project", self.save_project))
        self.smart_page.addLayout(header)
        self.editor_status = QLabel()
        self.editor_status.setStyleSheet("color: #aebed3;")
        self.statusBar().messageChanged.connect(self.editor_status.setText)
        self.smart_page.addWidget(self.editor_status)

        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(self.panel(preview))
        self.inspector = self.inspector_panel()
        self.inspector.setMinimumWidth(225)
        top.addWidget(self.inspector)
        self.advanced = self.scenes_tab()
        self.advanced.setMinimumWidth(275)
        top.addWidget(self.advanced)
        self.advanced.hide()
        top.setStretchFactor(0, 1)
        top.setSizes([760, 245, 300])
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(self.timeline_panel())
        split.setSizes([370, 235])
        self.smart_page.addWidget(split, 1)

    def inspector_panel(self):
        layout = QVBoxLayout()
        self.inspector_details = QLabel("Select a clip on the timeline.")
        self.inspector_details.setWordWrap(True)
        layout.addWidget(self.inspector_details)
        self.audio_mode = self.button("Switch audio mode", self.toggle_clip_audio)
        layout.addWidget(self.audio_mode)
        tools = QHBoxLayout()
        tools.addWidget(self.button("Preview", self.play_clip))
        self.replace_selected = self.button("Replace", self.replace_clip)
        self.delete_selected = self.button("Delete", self.delete_clip)
        tools.addWidget(self.replace_selected)
        tools.addWidget(self.delete_selected)
        layout.addLayout(tools)
        role = QHBoxLayout()
        role.addWidget(QLabel("Clip role"))
        self.clip_role = QComboBox()
        self.clip_role.addItems(["Scene", "Intro", "Outro"])
        self.clip_role.activated.connect(lambda _: self.perform(self.assign_role))
        role.addWidget(self.clip_role, 1)
        layout.addLayout(role)
        self.mix_inputs = {}
        for track, label in (("original", "Movie"), ("voice", "Narration"), ("music", "Music")):
            row = QHBoxLayout()
            row.addWidget(QLabel(label), 1)
            volume = QSpinBox()
            volume.setRange(0, 100)
            volume.setSuffix("%")
            volume.setValue(round(getattr(self.mix, track) * 100))
            volume.valueChanged.connect(lambda value, name=track: self.set_gain(name, value))
            self.mix_inputs[track] = volume
            row.addWidget(volume)
            if track != "original":
                row.addWidget(self.button("Import", lambda name=track: self.import_audio(name)))
                row.addWidget(self.button("Clear", lambda name=track: self.clear_audio(name)))
            layout.addLayout(row)
        layout.addStretch(1)
        return self.panel(layout)

    def scenes_tab(self):
        layout = QVBoxLayout()
        self.scene_search = QLineEdit()
        self.scene_search.setPlaceholderText("Source time or scene number")
        self.scene_search.textChanged.connect(self.filter_scenes)
        layout.addWidget(self.scene_search)
        self.scene_filter = QComboBox()
        self.scene_filter.addItems(["All scenes", "Selected", "Unused", "Original audio", "Recommended"])
        self.scene_filter.currentIndexChanged.connect(self.filter_scenes)
        layout.addWidget(self.scene_filter)
        self.scene_model = SceneModel(self)
        self.scene_grid = QListView()
        self.scene_grid.setViewMode(QListView.ViewMode.IconMode)
        self.scene_grid.setResizeMode(QListView.ResizeMode.Adjust)
        self.scene_grid.setLayoutMode(QListView.LayoutMode.Batched)
        self.scene_grid.setUniformItemSizes(True)
        self.scene_grid.setWordWrap(True)
        self.scene_grid.setIconSize(QSize(104, 58))
        self.scene_grid.setGridSize(QSize(132, 105))
        self.scene_grid.setModel(self.scene_model)
        self.scene_grid.selectionModel().currentChanged.connect(
            lambda index, _: self.perform(self.preview_candidate) if index.isValid() else None)
        layout.addWidget(self.scene_grid, 1)
        actions = QHBoxLayout()
        actions.addWidget(self.button("Add scene", self.add_candidate))
        actions.addWidget(self.button("Replace clip", self.replace_with_candidate))
        layout.addLayout(actions)
        self.highlight_controls = self.cut_tab()
        self.highlight_controls.hide()
        layout.addWidget(self.button("Analyze source only", self.analyze_source))
        layout.addWidget(self.button("Manual highlights", lambda: self.highlight_controls.setVisible(self.highlight_controls.isHidden())))
        layout.addWidget(self.highlight_controls)
        return self.panel(layout)

    def timeline_panel(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 3, 0, 0)
        toolbar = QHBoxLayout()
        self.playhead = QLabel("Montage timeline")
        self.playhead.setObjectName("previewTitle")
        toolbar.addWidget(self.playhead, 1)
        self.total_label = QLabel()
        toolbar.addWidget(self.total_label)
        for label, action in (("Undo", self.undo), ("Redo", self.redo), ("−", lambda: self.zoom(1 / 1.65)),
                              ("+", lambda: self.zoom(1.65)), ("Fit", lambda: self.zoom(None)),
                              ("Inspector", lambda: self.inspector.setVisible(self.inspector.isHidden())),
                              ("Scenes", lambda: self.advanced.setVisible(self.advanced.isHidden()))):
            toolbar.addWidget(self.button(label, action))
        layout.addLayout(toolbar)
        self.timeline = Timeline(self)
        self.timeline.seek.connect(self.seek_timeline)
        self.timeline.moved.connect(lambda a, b: self.perform(lambda: self.reorder(a, b)))
        self.timeline.preview.connect(lambda: self.perform(self.play_clip))
        self.timeline_scroll = QScrollArea()
        self.timeline_scroll.setWidgetResizable(True)
        self.timeline_scroll.setWidget(self.timeline)
        content = QHBoxLayout()
        content.setSpacing(0)
        self.track_labels = TrackLabels(self)
        content.addWidget(self.track_labels)
        content.addWidget(self.timeline_scroll, 1)
        layout.addLayout(content, 1)

        self.timeline.selected.connect(self.select_row)
        return self.panel(layout)

    def final_tab(self):
        layout = QVBoxLayout()
        self.final_summary = QLabel()
        self.final_summary.setWordWrap(True)
        layout.addWidget(self.final_summary)
        self.export_button = self.button("Validate and export MP4…", self.export)
        self.export_button.setObjectName("primary")
        layout.addWidget(self.export_button)
        layout.addWidget(self.button("Play assembly", lambda: self.preview(self.clips, assembly=True)))
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
        self.cancel_button.show()
        self.cancel_button.setEnabled(True)
        self.statusBar().showMessage(message)
        self.job = Job(operation)
        self.job.signals.progress.connect(self.statusBar().showMessage)

        def finish():
            self.tabs.setEnabled(True)
            self.progress.hide()
            self.cancel_button.hide()
            self.job = None
            self.statusBar().showMessage("Ready")

        def success(result):
            finish()
            self.perform(lambda: on_result(result))

        def failure(message):
            finish()
            if message:
                self.error(message)
            else:
                self.statusBar().showMessage("Operation cancelled")

        self.job.signals.result.connect(success)
        self.job.signals.error.connect(failure)
        QThreadPool.globalInstance().start(self.job)

    def cancel_operation(self):
        if self.job:
            self.job.cancel.set()
            self.cancel_button.setEnabled(False)
            self.statusBar().showMessage("Cancelling after the current processing step…")

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
        self.waves["original"] = AudioEvidence()
        self.undo_stack.clear()
        self.redo_stack.clear()
        if not self.title.text():
            self.title.setText(self.movie.path.stem)
        self.source_playback = False
        self.invalidate_preview()
        self.refresh()
        self.statusBar().showMessage("Movie imported. Choose a target and generate a review.")

    def import_audio(self, track):
        formats = "Audio (*.mp3 *.wav)" if track == "voice" else "Audio (*.mp3 *.wav *.aac *.m4a *.flac *.ogg)"
        path, _ = QFileDialog.getOpenFileName(self, f"Import {track}", "", formats)
        if path:
            def load(progress):
                media = inspect(Path(path), video=False)
                return media.path, audio_evidence(media.path, self.cache, progress)

            def apply(result):
                self.record_edit()
                setattr(self, track, result[0])
                self.waves[track] = result[1]
                self.invalidate_preview()
                self.refresh()

            self.background("Checking audio file…", load, apply)

    def clear_audio(self, track):
        self.record_edit()
        setattr(self, track, None)
        self.waves[track] = AudioEvidence()
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
        def load(progress):
            return analyze(self.movie, self.cache, progress), self.read_movie_evidence(progress)

        self.background("Analyzing movie…", load, self.set_analysis)

    def set_analysis(self, result):
        self.analysis, self.waves["original"] = result
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
        self.read_highlight_range(index, start, end)

    def read_highlight_range(self, index, start, end):
        self.background("Reading highlight frames…",
                        lambda _: fingerprint_highlight(self.movie, self.analysis, start, end, cache=self.cache),
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
        self.record_edit()
        self.highlights, self.clips = updated, edits
        self.invalidate_preview()
        self.refresh()
        self.highlight_list.setCurrentIndex(len(updated) - 1 if index is None else index)

    def preview_highlight(self):
        self.preview([self.highlights[self.highlight_index()]])

    def remove_highlight(self):
        index = self.highlight_index()
        self.remove_edit(self.highlights[index])

    def read_movie_evidence(self, progress):
        if not self.movie.has_audio:
            return AudioEvidence()
        executable, model = os.getenv("MRS_WHISPER_CLI"), os.getenv("MRS_WHISPER_MODEL")
        whisper = LocalWhisper(Path(executable), Path(model)) if executable and model else None
        try:
            return audio_evidence(self.movie.path, self.cache, progress, movie=True, transcriber=whisper)
        except (OSError, ValueError, RuntimeError) as error:
            if whisper:
                progress(f"Local transcription unavailable · {error}")
                try:
                    return audio_evidence(self.movie.path, self.cache, progress, movie=True)
                except (OSError, ValueError, RuntimeError):
                    pass
            progress("Audio analysis unavailable. Using visual shot analysis.")
            return AudioEvidence()

    def generate(self):
        self.require_movie()
        if self.clips and QMessageBox.question(
            self, "Regenerate montage", "Replace the current timeline with a new selection?"
        ) != QMessageBox.StandardButton.Yes:
            return
        target = self.target.value()
        self.tabs.setCurrentIndex(1)

        def build(progress):
            analysis = self.analysis or analyze(self.movie, self.cache, progress)
            evidence = self.read_movie_evidence(progress)
            highlights = self.highlights or suggest_highlights(self.movie, analysis, evidence, self.cache, progress)
            try:
                clips = assemble(analysis.candidates, highlights, self.movie.fps, analysis.frames,
                                 target, evidence, progress)
            except ValueError:
                if self.highlights or not highlights:
                    raise
                highlights = []
                clips = assemble(analysis.candidates, highlights, self.movie.fps, analysis.frames,
                                 target, evidence, progress)
            return analysis, evidence, highlights, clips

        def apply(result):
            self.record_edit()
            self.analysis, self.waves["original"], self.highlights, self.clips = result
            self.source_playback = False
            self.invalidate_preview()
            self.refresh()
            self.zoom(None)
            self.statusBar().showMessage("Review generated. Select a clip or play the complete montage.")

        self.background("Analyzing and generating the review…", build, apply)

    def snapshot(self):
        return (self.clips[:], self.highlights[:], self.mix, self.voice, self.music, self.waves.copy())

    def record_edit(self):
        self.undo_stack.append(self.snapshot())
        self.redo_stack.clear()

    def restore_edit(self, source, destination):
        if source:
            destination.append(self.snapshot())
            self.clips, self.highlights, self.mix, self.voice, self.music, waves = source.pop()
            self.waves.update({key: waves[key] for key in ("voice", "music")})
            self.sync_mix()
            self.invalidate_preview()
            self.refresh()

    def undo(self):
        self.restore_edit(self.undo_stack, self.redo_stack)

    def redo(self):
        self.restore_edit(self.redo_stack, self.undo_stack)

    def sync_mix(self):
        for key, widget in self.mix_inputs.items():
            widget.blockSignals(True)
            widget.setValue(round(getattr(self.mix, key) * 100))
            widget.blockSignals(False)

    def set_gain(self, name, value):
        if not self.job:
            self.record_edit()
            self.mix = replace(self.mix, **{name: value / 100})
            self.invalidate_preview()
            self.refresh()

    def set_clips(self, clips, *, record=True):
        if record:
            self.record_edit()
        self.clips = clips
        self.invalidate_preview()
        self.refresh()

    def zoom(self, ratio):
        self.timeline.zoom = min(64, max(1, self.timeline.zoom * ratio)) if ratio else 1
        self.timeline.update_width()
        cursor = self.timeline.frame
        if cursor is not None:
            self.timeline_scroll.ensureVisible(round(self.timeline.x_at(cursor)), 0, 80, 0)

    def assign_role(self):
        index = self.entries[self.selected_row()].edit_index
        updated = self.clips[:]
        clip = updated[index]
        role = ("", "Intro", "Outro")[self.clip_role.currentIndex()]
        if role != clip.role:
            self.record_edit()
            updated[index] = replace(clip, role=role)
            if clip.highlight:
                self.highlights[self.highlights.index(clip)] = updated[index]
            self.set_clips(updated, record=False)

    def toggle_clip_audio(self):
        self.require_movie(True)
        index = self.entries[self.selected_row()].edit_index
        clip = self.clips[index]
        changed = replace(clip, muted=not clip.muted) if clip.highlight else replace(clip, original=not clip.original)
        updated = self.clips[:]
        updated[index] = changed
        self.check(updated)
        self.record_edit()
        if clip.highlight:
            self.highlights[self.highlights.index(clip)] = changed
        self.set_clips(updated, record=False)
        self.select_edit(index)

    def reorder(self, start, end):
        old, new = self.entries[start].edit_index, self.entries[end].edit_index
        if old != new:
            updated = self.clips[:]
            updated.insert(new, updated.pop(old))
            self.check(updated)
            self.set_clips(updated)
            self.select_edit(new)

    def step_clip(self, delta):
        if not self.entries:
            raise ValueError("Generate a review first.")
        row = self.active_row if self.active_row is not None else self.selected_clip_row
        row = min(len(self.entries) - 1, max(0, row + delta))
        self.select_row(row)
        self.seek_timeline(self.entries[row].position / self.timeline.total)

    def check(self, clips, final=False):
        validate(clips, self.movie.fps, self.analysis.frames, final=final, has_audio=self.movie.has_audio)

    def selected_row(self):
        index = self.selected_clip_row
        if not 0 <= index < len(self.entries):
            raise ValueError("Select a timeline clip.")
        return index

    def play_clip(self):
        self.require_movie(True)
        row = self.selected_row()
        entry = self.entries[row]
        self.background("Preparing selected clip…", lambda progress: render_selection(
            self.movie, self.analysis, self.clips, row, self.voice, self.music, self.cache, progress, mix=self.mix),
            lambda path: self.load_player(path, f"Selected clip {row + 1}", montage=True, entry=entry))

    def select_edit(self, index):
        row = next((row for row, entry in enumerate(self.entries) if entry.edit_index == index), None)
        if row is not None:
            self.select_row(row)

    def select_row(self, row):
        self.selected_clip_row = row
        self.update_edit_controls()
        self.timeline_scroll.ensureVisible(round(self.timeline.x_at(self.entries[row].position)), 0, 80, 0)

    def update_edit_controls(self, *_):
        row = self.selected_clip_row
        entry = self.entries[row] if 0 <= row < len(self.entries) else None
        original = entry is not None and entry.segment.highlight
        self.replace_selected.setText("Replace highlight" if original else "Replace selected")
        self.delete_selected.setText("Trim" if original else "Delete")
        self.timeline.selection = row
        self.timeline.update()
        self.audio_mode.setEnabled(bool(entry) and self.movie.has_audio)
        self.clip_role.setEnabled(bool(entry))
        self.clip_role.setCurrentIndex({"Intro": 1, "Outro": 2}.get(entry.segment.role, 0) if entry else 0)
        if entry:
            fps = self.movie.fps
            clip = entry.segment
            mode = "Original movie audio" if clip.movie_audio else "Narration" if self.voice else "Narration placeholder"
            self.inspector_details.setText(
                f"Clip {row + 1}  ·  {mode}\n\n"
                f"Source  {clock(clip.start / fps)} – {clock(clip.end / fps)}\n"
                f"Montage  {clock(entry.position / fps)}\n"
                f"Duration  {float(clip.frames / fps):.3f} s  ·  {clip.frames} frames")
            self.audio_mode.setText("Use narration mode" if clip.movie_audio else "Use original audio")
        else:
            self.inspector_details.setText("Select a clip on the timeline.")
        self.replace_selected.setEnabled(bool(entry))
        self.delete_selected.setEnabled(bool(entry))
        if self.advanced.isVisible() and self.scene_filter.currentIndex() == 4:
            self.filter_scenes()

    def replace_clip(self):
        index = self.entries[self.selected_row()].edit_index
        clip = self.clips[index]
        if clip.highlight:
            self.highlight_list.setCurrentIndex(self.highlights.index(clip))
            self.start_time.setText(clock(clip.start / self.movie.fps))
            self.length.setValue(round(clip.frames / self.movie.fps))
            self.tabs.setCurrentIndex(1)
            self.advanced.show()
            self.highlight_controls.show()
            self.start_time.setFocus()
            self.statusBar().showMessage("Set a continuous source range, then click Replace in Cut.")
            return
        edits = self.clips.copy()
        edits[index] = replacement(edits, index, self.analysis.candidates)
        self.check(edits)
        self.set_clips(edits)
        self.select_edit(index)

    def delete_clip(self):
        entry = self.entries[self.selected_row()]
        clip = self.clips[entry.edit_index]
        if not clip.highlight:
            self.remove_edit(clip)
            return
        segment = entry.segment
        if segment.start != clip.start and segment.end != clip.end:
            raise ValueError("Trim a highlight from either end to keep the dialogue continuous.")
        start = segment.end if segment.start == clip.start else clip.start
        end = segment.start if segment.end == clip.end else clip.end
        if (end - start) / self.movie.fps < 10:
            raise ValueError("A highlight must keep at least 10 seconds. Use Cut to remove the entire highlight.")
        self.read_highlight_range(self.highlights.index(clip), start, end)

    def move_clip(self, delta):
        index = self.entries[self.selected_row()].edit_index
        destination = index + delta
        if 0 <= destination < len(self.clips):
            edits = self.clips.copy()
            edits[index], edits[destination] = edits[destination], edits[index]
            self.check(edits)
            self.set_clips(edits)
            self.select_edit(destination)

    def remove_edit(self, clip):
        edits = [item for item in self.clips if item != clip]
        if edits:
            self.check(edits)
        self.record_edit()
        self.highlights = [item for item in self.highlights if item != clip]
        self.set_clips(edits, record=False)

    def preview_source(self):
        self.require_movie()
        self.load_player(self.movie.path, "Source movie", source=True)

    def selected_candidate(self):
        self.require_movie(True)
        index = self.scene_grid.currentIndex()
        if not index.isValid():
            raise ValueError("Select a source scene.")
        return self.analysis.candidates[index.data(Qt.ItemDataRole.UserRole)]

    def preview_candidate(self):
        clip = self.selected_candidate()
        mode = any(overlaps(clip, group) for group in self.clips if group.movie_audio)
        self.load_player(self.movie.path, "Source scene", source=True,
                         position=max(0, round((clip.start / self.movie.fps + self.analysis.offset) * 1000)),
                         muted=not mode)
        self.browser_preview_token = round((clip.end / self.movie.fps + self.analysis.offset) * 1000)

    def add_candidate(self):
        edits = [*self.clips, self.selected_candidate()]
        self.check(edits)
        self.set_clips(edits)
        self.select_edit(len(edits) - 1)

    def replace_with_candidate(self):
        candidate = self.selected_candidate()
        index = self.entries[self.selected_row()].edit_index
        if self.clips[index].highlight:
            raise ValueError("Replace original-audio highlights with a continuous source range in Cut.")
        edits = self.clips.copy()
        edits[index] = candidate
        self.check(edits)
        self.set_clips(edits)
        self.select_edit(index)

    def preview(self, clips, *, assembly=False):
        self.require_movie(True)
        self.check(clips)
        voice = self.voice if assembly else None
        music = self.music if assembly else None
        label = "Montage preview" if assembly else "Source range preview"
        self.background("Preparing preview…", lambda progress: render(self.movie, self.analysis, clips, voice,
                        music, self.cache, progress, mix=self.mix if assembly else None), lambda path: self.load_player(
                        path, label, montage=assembly))

    def load_player(self, path, label, *, source=False, montage=False, entry=None, position=0, autoplay=True, muted=False):
        self.montage_playback = False
        self.pending_play = False
        self.browser_preview_token = None
        self.player.stop()
        self.audio.setMuted(muted)
        self.pause_on_frame = not autoplay
        if autoplay:
            self.video_stack.setCurrentWidget(self.video)
        elif entry:
            image = thumbnail(self.cache, self.source_key, entry.segment.start)
            self.still.setPixmap(image.scaled(self.video_stack.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation))
            self.video_stack.setCurrentWidget(self.still)
        self.pending_position = position
        self.source_playback = source
        self.montage_playback = montage
        self.play_entry = entry
        self.play_offset = entry.position if entry else 0
        self.set_active_clip(None)
        self.timeline.set_frame(self.play_offset if montage else None)
        self.position_changed(0)
        self.preview_label.setText(label)
        url = QUrl.fromLocalFile(str(path))
        self.pending_play = True
        if self.player.source() == url:
            self.start_pending_preview(self.player.mediaStatus())
        else:
            self.player.setSource(url)

    def start_pending_preview(self, status):
        if not self.pending_play or status not in (
            QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia,
            QMediaPlayer.MediaStatus.EndOfMedia,
        ):
            return
        self.pending_play = False
        self.player.setPosition(self.pending_position)
        self.player.play()

    def show_video_frame(self, frame):
        if self.pause_on_frame and frame.isValid() and not self.pending_play:
            if frame.startTime() >= 0 and frame.startTime() / 1000 + 1000 / self.movie.fps < self.pending_position:
                return
            self.pause_on_frame = False
            self.player.pause()
            image = frame.toImage()
            if not image.isNull():
                self.still.setPixmap(QPixmap.fromImage(image).scaled(
                    self.video_stack.size(), Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))

    def invalidate_preview(self):
        keep_source = self.source_playback and self.play_entry is None
        self.play_entry = None
        self.play_offset = 0
        self.montage_playback = False
        self.timeline.set_frame(None)
        self.set_active_clip(None)
        if not keep_source:
            self.pending_play = False
            self.pause_on_frame = False
            self.source_playback = False
            self.player.stop()
            self.player.setSource(QUrl())
            self.video_stack.setCurrentWidget(self.video)
            self.preview_label.setText("Preview")

    def toggle_playback(self):
        if self.clips and not self.montage_playback:
            self.preview(self.clips, assembly=True)
        elif not self.player.source().isValid():
            self.preview_source()
        elif self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def seek_timeline(self, fraction):
        if not self.movie or not self.entries:
            return
        frame = min(self.timeline.total - 1, max(0, round(fraction * self.timeline.total)))
        self.timeline.set_frame(frame)
        row = self.timeline.row_at(self.timeline.x_at(frame))
        self.set_active_clip(row)
        if self.montage_playback and self.play_entry is None:
            self.player.setPosition(round(frame * 1000 / self.movie.fps))
        elif self.play_entry and self.play_entry.position <= frame < self.play_entry.position + self.play_entry.segment.frames:
            self.player.setPosition(round((frame - self.play_entry.position) * 1000 / self.movie.fps))
        elif self.timeline.press is None or self.timeline.drag_row is not None:
            entry = self.entries[row]
            source_frame = entry.segment.start + frame - entry.position
            position = max(0, round((source_frame / self.movie.fps + self.analysis.offset) * 1000))
            self.load_player(self.movie.path, "Clip source frame", source=True, position=position,
                             autoplay=False, muted=True, entry=entry)
            self.timeline.set_frame(frame)
            self.set_active_clip(row)
        self.playhead.setText(f"Montage  ·  {clock(frame / self.movie.fps)}  ·  Clip {row + 1}")

    def seek_released(self):
        self.dragging = False
        self.player.setPosition(self.seek.sliderPosition())

    def seek_action(self, action):
        if not self.dragging:
            self.player.setPosition(self.seek.sliderPosition())

    def position_changed(self, position):
        if not self.dragging:
            self.seek.setValue(position)
        if self.browser_preview_token is not None and position >= self.browser_preview_token:
            self.browser_preview_token = None
            self.player.pause()
        self.position.setText(clock(position / 1000))
        if self.montage_playback:
            # Allow half a millisecond for Qt's rounded presentation timestamps.
            end = self.play_offset + self.play_entry.segment.frames if self.play_entry else self.timeline.total
            frame = self.play_offset + int((position * self.movie.fps + self.movie.fps / 2) / 1000)
            frame = min(frame, end - 1)
            self.timeline.set_frame(frame)
            for row, entry in enumerate(self.entries):
                if entry.position <= frame < entry.position + entry.segment.frames:
                    self.set_active_clip(row)
                    montage_time = self.play_offset / self.movie.fps + position / 1000
                    self.playhead.setText(f"Montage timeline  ·  {clock(montage_time)}  ·  Clip {row + 1}")
                    break

    def set_active_clip(self, row):
        if row == self.active_row:
            return
        self.active_row = row
        self.timeline.update()
        if row is None:
            self.playhead.setText("Montage timeline")
        elif self.montage_playback:
            self.timeline_scroll.ensureVisible(round(self.timeline.x_at(self.entries[row].position)), 0, 45, 0)

    def refresh_scenes(self):
        self.browser_source = self.analysis
        self.filter_scenes()

    def filter_scenes(self, *_):
        if not self.analysis:
            self.scene_model.update_rows([])
            return
        terms = self.scene_search.text().casefold().split()
        mode = self.scene_filter.currentIndex()
        used = {(clip.start, clip.end) for clip in self.clips}
        audible = [clip for clip in self.clips if clip.movie_audio]
        selected = self.selected_clip_row
        edit = self.entries[selected].edit_index if 0 <= selected < len(self.entries) else None
        rows = []
        for row, clip in enumerate(self.analysis.candidates):
            active = (clip.start, clip.end) in used
            original = any(overlaps(clip, group) for group in audible)
            status = "Original audio" if original else "Selected" if active else "Unused"
            if mode == 1 and not (active or original) or mode == 2 and (active or original) or mode == 3 and not original:
                continue
            if mode == 4:
                if edit is None or self.clips[edit].highlight or any(overlaps(clip, item) for item in self.clips):
                    continue
                if edit and adjacent_conflict(self.clips[edit - 1], clip):
                    continue
                if edit + 1 < len(self.clips) and adjacent_conflict(clip, self.clips[edit + 1]):
                    continue
            label = f"{row + 1}  ·  {clock(clip.start / self.movie.fps)}\n{float(clip.frames / self.movie.fps):.2f} s  ·  {status}"
            if all(term in label.casefold() for term in terms):
                rows.append((row, clip, label))
        if mode == 4:
            rows.sort(key=lambda item: (abs(item[1].frames - self.clips[edit].frames), -item[1].score))
        self.scene_model.update_rows(rows)

    def refresh_title(self):
        movie = self.movie
        self.editor_title.setText(f"{self.title.text() or movie.path.stem}   ·   {clock(movie.duration)} source"
                                  if movie else "No movie imported")

    def refresh(self):
        movie = self.movie
        self.movie_label.setText(f"{movie.path.name}\n{movie.width} × {movie.height} · {float(movie.fps):.3f} FPS · {clock(movie.duration)}"
                                 if movie else "No movie imported")
        self.refresh_title()
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
        entries = timeline_entries(self.clips, movie.fps) if movie else []
        if entries != self.entries or self.browser_source is not self.analysis:
            self.entries = entries
            self.timeline.set_clips([entry.segment for entry in entries])
            self.selected_clip_row = min(self.selected_clip_row, len(entries) - 1)
        self.update_edit_controls()
        self.refresh_scenes()
        cursor = sum(clip.frames for clip in self.clips)
        self.timeline.update_regions()
        self.track_labels.update()
        self.total_label.setText(f"{clock(cursor / movie.fps if movie else 0)} total")
        self.final_summary.setText(f"{len(entries)} video segments · {clock(cursor / movie.fps if movie else 0)}\n"
                                   f"{sum(clip.highlight and clip.movie_audio for clip in self.clips)} original-audio highlights in timeline\n"
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
            "version": 3, "mix": vars(self.mix), "movie": str(self.movie.path), "source_key": self.source_key,
            "voice": str(self.voice) if self.voice else None, "music": str(self.music) if self.music else None,
            "title": self.title.text(), "year": self.year.value(), "target": self.target.value(),
            "highlights": [[c.start, c.end, list(c.cuts), c.muted, c.role] for c in self.highlights],
            "clips": [[c.start, c.end, c.highlight, c.original, c.role] for c in self.clips],
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
            if payload["version"] not in (1, 2, 3):
                raise ValueError("Unsupported edit file version.")
            movie = inspect(Path(payload["movie"]))
            if identity(movie.path) != payload["source_key"]:
                raise ValueError("The saved edit references a different or changed movie file.")
            analysis = analyze(movie, self.cache, progress)
            highlights = [
                replace(fingerprint_highlight(movie, analysis, *item[:2], cache=self.cache,
                                              cuts=tuple(item[2]) if len(item) > 2 else None),
                        muted=item[3] if len(item) > 3 else False, role=item[4] if len(item) > 4 else "")
                for item in payload["highlights"]
            ]
            indexed = {(c.start, c.end, c.highlight): c for c in analysis.candidates + highlights}
            try:
                clips = [replace(indexed[tuple(item[:3])], original=bool(item[3]) if len(item) > 3 else False,
                                 role=item[4] if len(item) > 4 else "") for item in payload["clips"]]
            except KeyError as error:
                raise ValueError("A saved clip is missing from the source analysis.") from error
            if clips:
                validate(clips, movie.fps, analysis.frames, has_audio=movie.has_audio)
            ordered = sorted(highlights, key=lambda clip: clip.start)
            if len(ordered) > 3 or any(a.end > b.start for a, b in zip(ordered, ordered[1:])):
                raise ValueError("The saved highlights overlap or exceed the limit of three.")
            tracks = [Path(payload[name]) if payload[name] else None for name in ("voice", "music")]
            for track in tracks:
                if track:
                    inspect(track, video=False)
            waves = {name: audio_evidence(track, self.cache, progress) if track else AudioEvidence()
                     for name, track in zip(("voice", "music"), tracks)}
            try:
                waves["original"] = audio_evidence(movie.path, self.cache, progress, movie=True) if movie.has_audio else AudioEvidence()
            except (OSError, ValueError, RuntimeError):
                waves["original"] = AudioEvidence()
            return payload, movie, analysis, highlights, clips, tracks, waves

        def apply(result):
            payload, self.movie, self.analysis, self.highlights, self.clips, tracks, self.waves = result
            self.mix = Mix(**payload.get("mix", {}))
            self.sync_mix()
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.voice, self.music = tracks
            self.source_key = payload["source_key"]
            self.title.setText(payload["title"])
            self.year.setValue(payload["year"])
            self.target.setValue(payload["target"] if payload["version"] >= 3 else payload["target"] * 60)
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
            destination=destination, mix=self.mix), lambda output: QMessageBox.information(self, "Export complete", str(output)))


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
QListView, QScrollArea { background: #131c27; border: 1px solid #314156; }
QListView::item { padding: 3px; border-radius: 4px; }
QListView::item:selected { background: #2a5482; color: #ffffff; }
QSplitter::handle { background: #2d3c51; }
QStatusBar { color: #b5c5da; }
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
