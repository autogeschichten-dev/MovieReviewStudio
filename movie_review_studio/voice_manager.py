"""Compact editor for ordered narration slots and combined audio."""
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QPainter
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (QCheckBox, QDialog, QFileDialog, QFormLayout, QGridLayout,
                               QHBoxLayout, QLabel, QScrollArea, QSpinBox, QVBoxLayout, QWidget)


class Waveform(QLabel):
    def __init__(self, peaks):
        super().__init__()
        self.peaks = peaks
        self.setMinimumHeight(22)
        self.setAccessibleName("Audio waveform")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#192331"))
        painter.setPen(QColor("#7fcfef"))
        width = max(1, self.width())
        for x in range(width):
            first = x * len(self.peaks) // width
            end = (x + 1) * len(self.peaks) // width
            peak = max(self.peaks[first:max(first + 1, end)], default=0)
            height = max(1, round(peak * 10))
            painter.drawLine(x, 11 - height, x, 11 + height)


class Slot(QWidget):
    def __init__(self, insert):
        super().__init__()
        self.insert = insert
        self.setAcceptDrops(True)

    @staticmethod
    def paths(event):
        return [Path(url.toLocalFile()) for url in event.mimeData().urls()
                if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in {".wav", ".mp3"}]

    def dragEnterEvent(self, event):
        if self.paths(event):
            event.acceptProposedAction()

    def dropEvent(self, event):
        if paths := self.paths(event):
            event.acceptProposedAction()
            self.insert(paths)


class VoiceManager(QDialog):
    def __init__(self, studio, clock):
        super().__init__(studio)
        self.studio, self.clock = studio, clock
        self.setWindowTitle("Voice-over Manager")
        self.resize(740, 420)
        layout = QVBoxLayout(self)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        layout.addWidget(self.scroll, 1)
        settings = QFormLayout()
        self.pause = QSpinBox()
        self.pause.setRange(0, 5000)
        self.pause.setSingleStep(50)
        self.pause.setSuffix(" ms")
        settings.addRow("Pause between parts", self.pause)
        self.trim = QCheckBox("Trim sustained edge silence")
        settings.addRow(self.trim)
        self.fit = QCheckBox("Remove surplus picture clips to fit narration")
        self.fit.setToolTip("Preserves original-audio cuts and marked intro/outro clips.")
        settings.addRow(self.fit)
        self.captions = QLabel("Use part subtitles or configured local transcription")
        row = QHBoxLayout()
        row.addWidget(self.captions, 1)
        row.addWidget(studio.button("Choose SRT/VTT…", self.choose_captions))
        row.addWidget(studio.button("Auto", self.reset_captions))
        settings.addRow("Narration timing", row)
        layout.addLayout(settings)
        self.pause.valueChanged.connect(lambda _: self.update_options())
        self.trim.toggled.connect(lambda _: self.update_options())
        self.fit.toggled.connect(lambda _: self.update_options())
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        actions = QGridLayout()
        for index, (label, action) in enumerate((
            ("Add Audio Part", self.add_slot), ("Play All", self.play_all),
            ("Stop", self.stop), ("Export WAV / MP3…", self.export),
            ("Merge & Synchronize", self.synchronize), ("Clear Track", lambda: studio.clear_audio("voice")),
        )):
            button = studio.button(label, action)
            if label == "Merge & Synchronize":
                button.setObjectName("primary")
                self.sync_button = button
            actions.addWidget(button, index // 3, index % 3)
        layout.addLayout(actions)
        self.player = QMediaPlayer(self)
        self.output = QAudioOutput(self)
        self.player.setAudioOutput(self.output)
        self.player.errorOccurred.connect(lambda _, message: self.summary.setText(message))
        self.refresh()

    def hideEvent(self, event):
        self.stop()
        super().hideEvent(event)

    def stop(self):
        self.player.stop()

    def play(self, path):
        self.studio.player.pause()
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        self.player.play()

    def choose_captions(self):
        name, _ = QFileDialog.getOpenFileName(self, "Choose timing for the combined narration", "", "Subtitles (*.srt *.vtt)")
        if name:
            self.update_options(Path(name))

    def reset_captions(self):
        self.update_options(None)

    def update_options(self, captions=...):
        self.studio.voice_options(self.pause.value(), self.trim.isChecked(), self.fit.isChecked(), captions)

    def add_slot(self):
        self.studio.update_voice_slots((*self.studio.voice_parts, None))
        self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())

    def pick(self, index):
        files, _ = QFileDialog.getOpenFileNames(self, "Import ordered narration audio", "", "Audio (*.mp3 *.wav)")
        if files:
            self.studio.load_voice_parts(index, list(map(Path, files)))

    def remove(self, index):
        slots = list(self.studio.voice_parts)
        slots.pop(index)
        self.studio.update_voice_slots(tuple(slots))

    def move(self, index, step):
        target = index + step
        slots = list(self.studio.voice_parts)
        slots[index], slots[target] = slots[target], slots[index]
        self.studio.update_voice_slots(tuple(slots))

    def play_all(self):
        self.studio.process_voice("preview", self.play)

    def synchronize(self):
        self.stop()
        self.studio.process_voice("sync")

    def export(self):
        name, selected = QFileDialog.getSaveFileName(self, "Export combined narration", "Narration.wav",
                                                      "WAV audio (*.wav);;MP3 audio (*.mp3)")
        if not name:
            return
        path = Path(name)
        if not path.suffix:
            path = path.with_suffix(".mp3" if selected.startswith("MP3") else ".wav")
        if path.suffix.lower() not in {".wav", ".mp3"}:
            raise ValueError("Use a .wav or .mp3 filename.")
        self.studio.check_output_path(path)
        self.studio.process_voice("export", destination=path)

    def refresh(self):
        studio = self.studio
        for widget, value in zip((self.pause, self.trim, self.fit), (studio.voice_pause, studio.voice_trim, studio.voice_fit)):
            widget.blockSignals(True)
            if isinstance(widget, QSpinBox):
                widget.setValue(value)
            else:
                widget.setChecked(value)
            widget.blockSignals(False)
        self.captions.setText(studio.voice_captions.name if studio.voice_captions else
                              "Use part subtitles or configured local transcription")
        contents = QWidget()
        rows = QVBoxLayout(contents)
        for i, part in enumerate(studio.voice_parts):
            panel = Slot(lambda paths, index=i: studio.perform(lambda: studio.load_voice_parts(index, paths)))
            row = QGridLayout(panel)
            row.setContentsMargins(4, 4, 4, 4)
            row.addWidget(QLabel(f"Part {i + 1:02}"), 0, 0, 2, 1)
            name = QLabel(part.path.name if part else "Drop a WAV or MP3 here")
            name.setWordWrap(True)
            if part:
                name.setToolTip(str(part.path))
                row.addWidget(QLabel(self.clock(part.duration)), 0, 2)
                row.addWidget(Waveform(part.peaks), 1, 1, 1, 2)
            row.addWidget(name, 0, 1)
            row.setColumnStretch(1, 1)
            for column, (label, callback) in enumerate((
                ("Replace…" if part else "Import…", lambda index=i: self.pick(index)),
                ("Play", lambda item=part: self.play(item.path)),
                ("↑", lambda index=i: self.move(index, -1)),
                ("↓", lambda index=i: self.move(index, 1)),
                ("Delete", lambda index=i: self.remove(index)),
            ), 3):
                button = studio.button(label, callback)
                button.setEnabled(column == 3 or bool(part) and
                                  (column not in (5, 6) or 0 <= i + (1 if column == 6 else -1) < len(studio.voice_parts)))
                row.addWidget(button, 0, column)
            rows.addWidget(panel)
        rows.addStretch()
        scroll = self.scroll.verticalScrollBar().value()
        self.scroll.setWidget(contents)
        self.scroll.verticalScrollBar().setValue(scroll)
        self.sync_button.setEnabled(bool(studio.clips) and any(studio.voice_parts))
        self.update_summary()

    def update_summary(self):
        studio = self.studio
        target = studio.target.value()
        parts = [part for part in studio.voice_parts if part]
        estimate = sum(part.duration for part in parts) + max(0, len(parts) - 1) * studio.voice_pause / 1000
        recording = studio.voice_recording
        duration = recording.duration if recording else estimate
        difference = duration - target
        lines = [
            f"Total narration {self.clock(duration)}{' (estimate before processing)' if parts and not recording else ''}",
            f"Target video {self.clock(target)}",
            f"Difference {'+' if difference >= 0 else '−'}{self.clock(abs(difference))}",
        ]
        if studio.movie and studio.clips:
            available = sum(c.frames for c in studio.clips if not c.movie_audio) / studio.movie.fps
            picture = sum(c.frames for c in studio.clips) / studio.movie.fps
            lines.append(f"Current picture {self.clock(picture)} · narration capacity {self.clock(available)}")
        if studio.voice_sync and not studio.voice_sync_valid():
            lines.append("The timeline changed. Run Merge & Synchronize again.")
        if studio.voice_sync and studio.voice_sync_valid():
            lines.append("Script timing matched" if studio.voice_sync["aligned"] else "Synced to existing cuts using audio pauses")
        self.summary.setText("   ·   ".join(lines))
