"""Qt desktop workflow for importing, editing, previewing and exporting one review."""
from dataclasses import replace
from pathlib import Path
from threading import Event
import os
import re
from time import monotonic
import json
import logging
import math
import sys

from PySide6.QtCore import QObject, QRunnable, QSize, QStandardPaths, QThreadPool, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut, QValidator
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFormLayout, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QListView, QMainWindow, QMessageBox, QProgressBar, QScrollArea,
    QPushButton, QSlider, QSpinBox, QSplitter, QStackedWidget, QTabWidget, QPlainTextEdit,
    QVBoxLayout, QWidget,
)

from .analysis import analyze, atomic_json, fingerprint_highlight
from .media import Cancelled, cancellation, identity, inspect, invoke
from . import director, local_ai
from .ai_settings import AISettingsDialog, model_choices, settings
from .editor import SceneModel, Timeline, TrackLabels, thumbnail
from .model import adjacent_conflict, overlaps, replacement, timeline_entries, validate
from .render import Mix, render, render_selection
from .smart import AudioEvidence, LocalWhisper, assemble, audio_evidence, suggest_highlights


class JobSignals(QObject):
    progress = Signal(str)
    observation = Signal(object)
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
        if isinstance(message, dict):
            self.signals.observation.emit(message)
        else:
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
        self.script = []
        self.script_approved = False
        self.director_language = "English"
        self.director_mode = director.MODES[0]
        self.generated_voice = None
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
        observation = QHBoxLayout()
        self.vision_image = QLabel()
        self.vision_image.setFixedSize(160, 90)
        self.vision_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        observation.addWidget(self.vision_image)
        self.vision_text = QPlainTextEdit()
        self.vision_text.setReadOnly(True)
        self.vision_text.setFixedHeight(90)
        observation.addWidget(self.vision_text, 1)
        dismiss = QPushButton("Hide")
        dismiss.clicked.connect(lambda: self.vision_panel.hide())
        observation.addWidget(dismiss)
        self.vision_panel = self.panel(observation)
        self.vision_panel.hide()
        layout.addWidget(self.vision_panel)
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
        self.analyze_button = self.button("Fast Montage", self.generate)
        self.analyze_button.setObjectName("primary")
        self.import_target = TargetDuration()
        self.import_target.setRange(1020, 1080)
        self.import_target.setValue(1050)
        self.import_target.setSingleStep(5)
        form.addRow("Target duration", self.import_target)
        form.addRow(self.analyze_button)
        form.addRow(self.button("Analyze Movie / Generate AI Review…", self.open_director))
        form.addRow(self.button("AI Settings…", self.open_ai_settings))
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
        generate = self.button("Fast Montage", self.generate)
        generate.setObjectName("primary")
        header.addWidget(generate)
        header.addWidget(self.button("AI Director…", self.open_director))
        header.addWidget(self.button("AI Settings…", self.open_ai_settings))
        header.addWidget(self.button("Save project", self.save_project))
        self.smart_page.addLayout(header)
        self.editor_status = QLabel()
        self.editor_status.setStyleSheet("color: #aebed3;")
        self.statusBar().messageChanged.connect(self.editor_status.setText)
        self.smart_page.addWidget(self.editor_status)

        top = QSplitter(Qt.Orientation.Horizontal)
        self.editor_split = top
        top.addWidget(self.panel(preview))
        self.inspector = self.inspector_panel()
        self.inspector.setMinimumWidth(225)
        top.addWidget(self.inspector)
        self.advanced = self.scenes_tab()
        self.advanced.setMinimumWidth(275)
        top.addWidget(self.advanced)
        self.script_panel = self.build_script_panel()
        top.addWidget(self.script_panel)
        self.script_panel.hide()
        self.advanced.hide()
        top.setStretchFactor(0, 1)
        top.setSizes([760, 245, 300])
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(self.timeline_panel())
        split.setSizes([370, 235])
        self.smart_page.addWidget(split, 1)

    def show_script_panel(self, visible):
        self.script_panel.setVisible(visible)
        if visible:
            sizes = self.editor_split.sizes()
            sizes[3] = 340
            sizes[0] = max(400, sizes[0] - 270)
            self.editor_split.setSizes(sizes)

    def build_script_panel(self):
        layout = QVBoxLayout()
        self.script_picker = QComboBox()
        self.script_picker.activated.connect(self.seek_script)
        layout.addWidget(self.script_picker)
        self.script_text = QPlainTextEdit()
        self.script_text.setPlaceholderText("Select a narration section")
        layout.addWidget(self.script_text, 1)
        self.script_evidence = QLabel()
        self.script_evidence.setWordWrap(True)
        layout.addWidget(self.script_evidence)
        actions = QHBoxLayout()
        actions.addWidget(self.button("Apply text", self.apply_script_text))
        actions.addWidget(self.button("Approve", self.approve_script))
        actions.addWidget(self.button("Make voice", self.make_script_voice))
        actions.addWidget(self.button("Link clip", self.link_script_clip))
        layout.addLayout(actions)
        layout.addWidget(self.button("Export timeline JSON…", self.export_director_timeline))
        layout.addWidget(self.button("Import timeline JSON…", self.import_director_timeline))
        layout.addWidget(self.button("Align imported narration…", self.align_script_voice))
        panel = self.panel(layout)
        panel.setMinimumWidth(255)
        return panel

    def open_ai_settings(self):
        AISettingsDialog(self, Job).exec()

    def show_observation(self, row):
        if row["image"]:
            pixmap = QPixmap()
            if pixmap.loadFromData(row["image"]):
                self.vision_image.setPixmap(pixmap.scaled(self.vision_image.size(),
                    Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.vision_image.clear()
            self.vision_image.setText("No saved still")
        self.vision_text.setPlainText(
            f"Visual analysis {row['processed']}/{row['total']} · {row['successful']} successful · "
            f"{row['failed']} skipped/failed · source {clock(row['seconds'])}\n"
            f"Elapsed {clock(row['elapsed'])} · ETA {clock(row['eta'])}\n"
            f"{row['description']}")
        self.vision_panel.show()

    def open_director(self):
        from PySide6.QtWidgets import QCheckBox, QDialog, QDialogButtonBox
        self.require_movie()
        dialog = QDialog(self)
        dialog.setWindowTitle("Analyze Movie / Generate AI Review")
        dialog.resize(540, 535)
        action = "generate"
        form = QFormLayout(dialog)
        mode = QComboBox()
        mode.addItems(director.MODES)
        mode.setCurrentText(self.director_mode)
        form.addRow("Content mode", mode)
        language = QComboBox()
        language.addItems(list(director.LANGUAGES))
        language.setCurrentText(self.director_language)
        form.addRow("Narration language", language)
        highlights_enabled = QCheckBox("Include suggested original-audio highlights")
        highlights_enabled.setChecked(True)
        form.addRow(highlights_enabled)
        connection = QLabel("Checking Ollama at 127.0.0.1:11434…")
        form.addRow("Ollama", connection)
        visual = QComboBox()
        stored = settings()
        visual.addItem(stored.value("vision", local_ai.DEFAULT_MODEL, str))
        form.addRow("Visual model", visual)
        reasoning = QComboBox()
        reasoning.setEditable(True)
        reasoning.setCurrentText(stored.value("text", os.getenv("MRS_TEXT_MODEL", "qwen2.5:3b"), str))
        form.addRow("Text reasoning model", reasoning)
        whisper = QComboBox()
        whisper.addItems(["base", "small"])
        whisper.setCurrentText(stored.value("speech", "base", str))
        whisper.currentTextChanged.connect(lambda value: AISettingsDialog.store("speech", value))
        form.addRow("Speech model (CPU)", whisper)
        def list_models(result):
            version, installed = result
            connection.setText(f"Connected · {version} · {len(installed)} models")
            chosen = visual.currentText(), reasoning.currentText()
            for combo, saved, names in zip((visual, reasoning), chosen, model_choices(installed, *chosen)):
                combo.clear()
                combo.addItems(names)
                combo.setCurrentText(saved if saved in names else names[0])
        discovery = Job(lambda _: local_ai.discover())
        discovery.signals.result.connect(lambda result: list_models(result) if dialog.isVisible() else None)
        discovery.signals.error.connect(lambda _: connection.setText("Unavailable · start Ollama locally") if dialog.isVisible() else None)
        QTimer.singleShot(0, lambda: QThreadPool.globalInstance().start(discovery))
        code = QPlainTextEdit()
        code.setPlaceholderText("Paste original commentary for Custom Script")
        form.addRow(code)

        def load_text():
            name, _ = QFileDialog.getOpenFileName(dialog, "Import a script", "", "Text (*.txt)")
            if name:
                try:
                    code.setPlainText(Path(name).read_text(encoding="utf-8-sig"))
                    mode.setCurrentText("Custom Script")
                except (OSError, UnicodeError) as error:
                    self.error(str(error))

        form.addRow(self.button("Import TXT…", load_text))
        disclosure = QLabel("Analyze Movie uses local Moondream and local subtitles or faster-whisper. "
                            "Generate AI Review uses a separate text model. A configured cloud planner "
                            "receives descriptions and transcript excerpts only, with your approval.")
        disclosure.setWordWrap(True)
        form.addRow(disclosure)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        box.button(QDialogButtonBox.StandardButton.Ok).setText("Generate AI Review")
        analyze_only = box.addButton("Analyze Movie", QDialogButtonBox.ButtonRole.ActionRole)
        def start_analysis():
            nonlocal action
            action = "analyze"
            dialog.accept()
        analyze_only.clicked.connect(start_analysis)
        box.accepted.connect(dialog.accept)
        box.rejected.connect(dialog.reject)
        form.addRow(box)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        provider = director.VisionProvider.from_environment(reasoning.currentText(), text_only=True) if action == "generate" else None
        visual_model, speech_model = visual.currentText(), whisper.currentText()
        AISettingsDialog.store("vision", visual_model)
        AISettingsDialog.store("text", reasoning.currentText())
        self.vision_panel.hide()
        if provider and provider.cloud:
            consent = QMessageBox.question(self, "Allow cloud processing?",
                f"Send scene descriptions, transcript excerpts and script text to {provider.url}? "
                "Your provider may charge for this request.")
            if consent != QMessageBox.StandardButton.Yes:
                return
        content_mode, lang, text = mode.currentText(), language.currentText(), code.toPlainText()
        target = self.target.value()
        include_highlights = highlights_enabled.isChecked()

        def work(progress):
            progress("Scene detection · checking movie analysis")
            analysis = self.analysis or analyze(self.movie, self.cache, progress)
            _, models = local_ai.discover()
            if visual_model not in models or "moondream" not in visual_model.lower():
                raise ValueError(f"Install {visual_model} in Ollama before running visual analysis.")
            transcript = local_ai.dialogue(self.movie, self.cache, progress, speech_model)
            audio = (audio_evidence(self.movie.path, self.cache, progress, movie=True)
                     if self.movie.has_audio else AudioEvidence())
            cues = transcript["segments"]
            if cues:
                audio = replace(audio, captions=tuple((float(c["start"]), float(c["end"])) for c in cues))
            highlights = (self.highlights or suggest_highlights(self.movie, analysis, audio, self.cache, progress)
                          ) if include_highlights else []
            local_planner = provider and provider.url == local_ai.OLLAMA + "/v1/chat/completions"
            if local_planner:
                local_ai.release_model(provider.model)
            vision = local_ai.VisionBatch(self.movie, analysis, self.cache, visual_model, models[visual_model])
            def observations(clips, report):
                try:
                    result = vision.describe(clips, report, observe=report)
                except Exception:
                    try:
                        vision.unload()
                    except RuntimeError:
                        logging.warning("Could not unload the vision model after failure.")
                    raise
                vision.unload()
                return result
            if action == "analyze":
                candidates = [clip for clip in analysis.candidates if not clip.production_card]
                selected = [clip for group in director.chapters(candidates, 16) for clip in director.sample_group(group)]
                count = len(observations(selected, progress))
                return analysis, audio, count, transcript["provider"], vision.failed
            try:
                clips, script = director.plan(self.movie, analysis, highlights, provider, content_mode, lang,
                                              target, text, self.cache, progress, observations=observations,
                                              transcript=cues)
            except Exception:
                if local_planner:
                    try:
                        local_ai.release_model(provider.model)
                    except RuntimeError:
                        logging.warning("Could not unload the text model after failure.")
                raise
            if local_planner:
                local_ai.release_model(provider.model)
            return analysis, audio, highlights, clips, script, vision.failed

        def apply(result):
            if action == "analyze":
                self.analysis, self.waves["original"], count, transcript_source, failed = result
                self.refresh()
                self.statusBar().showMessage(f"Analysis cached · {count} scenes · {failed} skipped/failed · dialogue: {transcript_source}")
                return
            self.record_edit()
            self.drop_generated_voice()
            self.analysis, self.waves["original"], self.highlights, self.clips, self.script, failed = result
            self.director_mode, self.director_language = content_mode, lang
            self.script_approved = False
            self.source_playback = False
            self.invalidate_preview()
            self.tabs.setCurrentIndex(1)
            self.show_script_panel(True)
            self.refresh()
            self.statusBar().showMessage(f"AI draft ready · {failed} skipped or failed scenes · review evidence and script.")

        self.background("Preparing local analysis…" if action == "analyze" else "Preparing grounded AI review…", work, apply)

    def refresh_script(self, selected=None):
        current = self.script_picker.currentIndex()
        self.script_picker.clear()
        linked = director.placements(self.script, self.clips, self.movie.fps) if self.movie else {}
        for section in self.script:
            ranges = linked.get(section.id, [])
            where = clock(ranges[0]["montage_in"]) if ranges else "Unassigned"
            self.script_picker.addItem(f"{section.id} · {where}", section.id)
        if self.script:
            index = selected if selected is not None else min(max(0, current), len(self.script) - 1)
            self.script_picker.setCurrentIndex(index)
            section = self.script[index]
            self.script_text.setPlainText(section.text)
            ranges = linked.get(section.id, [])
            details = (f"{len(ranges)} linked segments · " +
                       (f"{section.duration:.2f} s recorded" if section.duration is not None else "No measured narration"))
            if ranges:
                first = ranges[0]
                details += f"\nSource {clock(first['source_in'])} – {clock(first['source_out'])}"
                if section.evidence:
                    details += "\n" + section.evidence[0][2][:175]
            self.script_evidence.setText(details)
        else:
            self.script_text.clear()
            self.script_evidence.clear()

    def seek_script(self, index):
        self.refresh_script(index)
        section = self.script[index]
        linked = director.placements([section], self.clips, self.movie.fps)[section.id]
        if linked and self.timeline.total:
            self.seek_timeline(linked[0]["montage_in"] * float(self.movie.fps) / self.timeline.total)

    def apply_script_text(self):
        index = self.script_picker.currentIndex()
        text = self.script_text.toPlainText().strip()
        if index < 0 or not text:
            raise ValueError("Select a section and enter its narration text.")
        if text != self.script[index].text:
            self.record_edit()
            self.script[index] = replace(self.script[index], text=text, duration=None)
            self.script_approved = False
            self.drop_generated_voice()
            self.invalidate_preview()
            self.refresh_script(index)

    def link_script_clip(self):
        from PySide6.QtWidgets import QInputDialog
        index = self.script_picker.currentIndex()
        if index < 0:
            raise ValueError("Select a script section first.")
        clip = self.entries[self.selected_row()].segment
        if clip.movie_audio or clip.highlight:
            raise ValueError("Select muted narration footage to link to the script.")
        observation, accepted = QInputDialog.getText(self, "Link source evidence",
                                                      "What visible or spoken evidence supports this section?")
        observation = observation.strip()
        if not accepted or not observation:
            return
        self.record_edit()
        for position, section in enumerate(self.script):
            evidence = tuple(item for item in section.evidence if item[:2] != (clip.start, clip.end))
            if position == index:
                evidence += ((clip.start, clip.end, observation[:1200]),)
            if evidence != section.evidence:
                self.script[position] = replace(section, evidence=evidence, duration=None)
        self.script_approved = False
        self.drop_generated_voice()
        self.invalidate_preview()
        self.refresh_script(index)

    def approve_script(self):
        if not self.script:
            raise ValueError("Generate a script first.")
        self.apply_script_text()
        self.record_edit()
        self.script_approved = True
        self.statusBar().showMessage("Script approved.")

    def drop_generated_voice(self):
        if self.generated_voice and self.voice == self.generated_voice:
            self.voice = None
            self.waves["voice"] = AudioEvidence()
        self.generated_voice = None

    def make_script_voice(self):
        if not self.script or not self.script_approved or self.script_text.toPlainText().strip() != self.script[self.script_picker.currentIndex()].text:
            raise ValueError("Apply text changes and approve the script before generating narration.")

        def work(progress):
            audio, script = director.synthesize(self.script, self.clips, self.movie,
                                                self.director_language, self.cache, progress)
            return audio, script, audio_evidence(audio, self.cache, progress)

        def apply(result):
            self.record_edit()
            self.voice, self.script, self.waves["voice"] = result
            self.generated_voice = self.voice
            self.invalidate_preview()
            self.refresh()

        self.background("Generating narration with configured offline TTS…", work, apply)

    def align_script_voice(self):
        if not self.voice or not self.script or not self.script_approved:
            raise ValueError("Import narration and approve the script before transcript alignment.")
        if self.generated_voice == self.voice:
            raise ValueError("Select the original imported recording before aligning again.")

        def work(progress):
            audio, script = director.align_recording(self.voice, self.script, self.clips, self.movie,
                                                      self.director_language, self.cache, progress)
            return audio, script, audio_evidence(audio, self.cache, progress)

        def apply(result):
            self.record_edit()
            self.voice, self.script, self.waves["voice"] = result
            self.generated_voice = self.voice
            self.invalidate_preview()
            self.refresh()

        self.background("Aligning local narration transcript…", work, apply)

    def export_director_timeline(self):
        self.require_movie(True)
        self.check(self.clips)
        path, _ = QFileDialog.getSaveFileName(self, "Export timeline", "timeline.json", "JSON (*.json)")
        if not path:
            return
        if Path(path).resolve() in {item.resolve() for item in (self.movie.path, self.voice, self.music) if item}:
            raise ValueError("Save timeline JSON to a different file from the imported media.")
        atomic_json(Path(path), director.timeline_document(self.movie, self.clips, self.script))

    def import_director_timeline(self):
        self.require_movie(True)
        path, _ = QFileDialog.getOpenFileName(self, "Import timeline", "", "JSON (*.json)")
        if path:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            clips = director.import_timeline(payload, self.movie, self.analysis, self.cache)
            self.record_edit()
            self.script = []
            self.script_approved = False
            self.highlights = [clip for clip in clips if clip.highlight]
            self.drop_generated_voice()
            self.set_clips(clips, record=False)

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
                              ("Scenes", lambda: self.advanced.setVisible(self.advanced.isHidden())),
                              ("Script", lambda: self.show_script_panel(self.script_panel.isHidden()))):
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
        layout.addWidget(self.button("Check exported video…", self.check_recap_export))
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
        started = monotonic()
        stage_start = started
        stage = ""
        recent = message
        estimate = None
        timer = QTimer(self)

        def show_time():
            elapsed = clock(monotonic() - started).split(".")[0]
            left = "ETA estimating" if estimate is None else "ETA " + clock(max(0, estimate - monotonic())).split(".")[0]
            if self.cancel_button.isEnabled():
                self.statusBar().showMessage(f"{recent}  ·  {elapsed} elapsed  ·  {left}")

        def update_progress(text):
            nonlocal stage, stage_start, recent, estimate
            recent = text
            numbered = re.search(r"(\d+)\s*(?:/|of)\s*(\d+)", text)
            if numbered:
                key = text[:numbered.start()].strip()
                if key != stage:
                    stage, stage_start = key, monotonic()
                done, total = map(int, numbered.groups())
                if done > 1:
                    estimate = monotonic() + (monotonic() - stage_start) * (total - done) / (done - 1)
            elif not text.startswith("Waiting for"):
                stage, estimate = "", None
            show_time()
        self.job.signals.progress.connect(update_progress)
        self.job.signals.observation.connect(self.show_observation)
        timer.timeout.connect(show_time)
        timer.start(1000)

        def finish():
            timer.stop()
            timer.deleteLater()
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
        self.script = []
        self.script_approved = False
        self.drop_generated_voice()
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
        if edits != self.clips:
            self.script_approved = False
            self.drop_generated_voice()
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
            self.script = []
            self.script_approved = False
            self.drop_generated_voice()
            self.analysis, self.waves["original"], self.highlights, self.clips = result
            self.source_playback = False
            self.invalidate_preview()
            self.refresh()
            self.zoom(None)
            self.statusBar().showMessage("Review generated. Select a clip or play the complete montage.")

        self.background("Analyzing and generating the review…", build, apply)

    def snapshot(self):
        return (self.clips[:], self.highlights[:], self.mix, self.voice, self.music, self.waves.copy(),
                self.script[:], self.script_approved, self.director_language, self.director_mode, self.generated_voice)

    def record_edit(self):
        self.undo_stack.append(self.snapshot())
        self.redo_stack.clear()

    def restore_edit(self, source, destination):
        if source:
            destination.append(self.snapshot())
            (self.clips, self.highlights, self.mix, self.voice, self.music, waves, self.script,
             self.script_approved, self.director_language, self.director_mode, self.generated_voice) = source.pop()
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
        if self.script and clips != self.clips:
            self.script_approved = False
            self.drop_generated_voice()
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
        if self.script_panel.isVisible() and self.script:
            clip = self.entries[row].segment
            for index, section in enumerate(self.script):
                if any((start, end) == (clip.start, clip.end) for start, end, _ in section.evidence):
                    self.refresh_script(index)
                    break
        self.timeline_scroll.ensureVisible(round(self.timeline.x_at(self.entries[row].position)), 0, 80, 0)

    def update_edit_controls(self, *_):
        row = self.selected_clip_row
        entry = self.entries[row] if 0 <= row < len(self.entries) else None
        original = entry is not None and entry.segment.highlight
        self.replace_selected.setText("Replace group" if original else "Replace")
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
        self.refresh_script()
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
        linked = director.placements(self.script, self.clips, self.movie.fps)
        atomic_json(destination, {
            "version": 4, "director": {"sections": [dict(vars(s), placements=linked[s.id]) for s in self.script],
                                        "approved": self.script_approved, "mode": self.director_mode,
                                        "language": self.director_language, "generated_voice": str(self.generated_voice) if self.generated_voice else None},
            "mix": vars(self.mix), "movie": str(self.movie.path), "source_key": self.source_key,
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
            if payload["version"] not in (1, 2, 3, 4):
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
            saved = payload.get("director", {})
            self.script = [director.ScriptSection.restore(section) for section in saved.get("sections", [])]
            self.script_approved = saved.get("approved", False)
            self.director_language = saved.get("language", "English")
            self.director_mode = saved.get("mode", director.MODES[0])
            self.generated_voice = Path(saved["generated_voice"]) if saved.get("generated_voice") else None
            self.show_script_panel(bool(self.script))
            self.source_key = payload["source_key"]
            self.title.setText(payload["title"])
            self.year.setValue(payload["year"])
            self.target.setValue(payload["target"] if payload["version"] >= 3 else payload["target"] * 60)
            self.source_playback = False
            self.invalidate_preview()
            self.refresh()
            self.tabs.setCurrentIndex(1)

        self.background("Opening saved edit…", load, apply)

    def check_recap_export(self):
        self.require_movie(True)
        self.check(self.clips)
        path, _ = QFileDialog.getOpenFileName(self, "Check exported video", "", "MP4 video (*.mp4)")
        if not path:
            return
        def work(progress):
            progress("Running Video Recap Skills quality checks…")
            report = director.quality_report(Path(path), self.movie, self.clips, self.script, self.voice, self.music)
            output = Path(path).with_suffix(".check.json")
            atomic_json(output, report)
            return output, report.get("ok")

        def done(result):
            output, ok = result
            label = "Checks passed; review any warnings." if ok else "Review the reported issues."
            QMessageBox.information(self, "Video quality report", f"{label}\n{output}")

        self.background("Checking exported video…", work, done)

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
