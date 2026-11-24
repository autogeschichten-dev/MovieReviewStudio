"""Local Ollama model selection and connection checks."""
from PySide6.QtCore import QSettings, QThreadPool, QTimer
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFormLayout, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
)

from . import ai_diagnostic, local_ai


def settings():
    return QSettings("MovieReviewStudio", "LocalAI")


def model_choices(installed, vision, text):
    visual = [name for name in installed if "moondream" in name.casefold()]
    planners = [name for name in installed if "moondream" not in name.casefold()]
    return (visual or [vision or local_ai.DEFAULT_MODEL], planners or [text or "qwen2.5:3b"])


class AISettingsDialog(QDialog):
    def __init__(self, parent, job_type):
        super().__init__(parent)
        self.job_type = job_type
        self.job = None
        self.setWindowTitle("AI Settings")
        self.resize(640, 435)
        layout = QFormLayout(self)
        self.connection = QLabel("Detecting the local Ollama service…")
        layout.addRow("Ollama", self.connection)
        self.hardware = QLabel("Detecting hardware…")
        self.hardware.setWordWrap(True)
        layout.addRow("Hardware", self.hardware)
        self.vision = QComboBox()
        self.text = QComboBox()
        saved = settings()
        self.vision.addItem(saved.value("vision", local_ai.DEFAULT_MODEL, str))
        self.text.addItem(saved.value("text", "qwen2.5:3b", str))
        layout.addRow("Vision model", self.vision)
        layout.addRow("Text model", self.text)
        self.vision.currentTextChanged.connect(lambda name: self.store("vision", name))
        self.text.currentTextChanged.connect(lambda name: self.store("text", name))
        self.buttons = []
        row = QHBoxLayout()
        for title, kind in (("Test Vision AI", "vision"), ("Test Text AI", "text"),
                            ("Test All Connections", "all")):
            button = QPushButton(title)
            button.clicked.connect(lambda _checked=False, selected=kind: self.test(selected))
            self.buttons.append(button)
            row.addWidget(button)
        layout.addRow(row)
        self.results = QPlainTextEdit()
        self.results.setReadOnly(True)
        self.results.setPlaceholderText("Connection test results appear here.")
        layout.addRow(self.results)
        done = QPushButton("Done")
        done.clicked.connect(self.accept)
        layout.addRow(done)
        QTimer.singleShot(0, self.detect)

    @staticmethod
    def store(key, value):
        if value:
            saved = settings()
            saved.setValue(key, value)
            saved.sync()

    def reject(self):
        if self.job is None:
            super().reject()

    def accept(self):
        if self.job is None:
            super().accept()

    def execute(self, operation, done, failed=None):
        if self.job:
            return
        for button in self.buttons:
            button.setEnabled(False)
        self.job = self.job_type(operation)

        def complete(value, callback):
            self.job = None
            for button in self.buttons:
                button.setEnabled(True)
            callback(value)

        self.job.signals.result.connect(lambda value: complete(value, done))
        self.job.signals.error.connect(lambda error: complete(error, failed or self.results.setPlainText))
        QThreadPool.globalInstance().start(self.job)

    def detect(self):
        def work(_progress):
            try:
                status = local_ai.discover()
            except RuntimeError:
                status = None
            return status, ai_diagnostic.hardware_summary()

        def done(result):
            status, hardware = result
            self.hardware.setText(hardware)
            if status is None:
                self.connection.setText("Unavailable · start Ollama on 127.0.0.1:11434")
                return
            version, installed = status
            self.connection.setText(f"Connected · Ollama {version} · {len(installed)} installed models")
            old = self.vision.currentText(), self.text.currentText()
            for combo, key, saved, names in zip((self.vision, self.text), ("vision", "text"), old,
                                                model_choices(installed, *old)):
                combo.blockSignals(True)
                combo.clear()
                combo.addItems(names)
                combo.setCurrentText(saved if saved in names else names[0])
                combo.blockSignals(False)
                self.store(key, combo.currentText())
        self.execute(work, done)

    def test(self, kind):
        self.results.setPlainText("Running local test…")
        vision, text = self.vision.currentText(), self.text.currentText()

        def work(progress):
            version, installed = local_ai.discover()
            results = [f"Local Ollama {version} · {len(installed)} installed models"]
            if kind in ("vision", "all"):
                if vision not in installed:
                    raise ValueError(f"Install the selected vision model: {vision}")
                local_ai.release_model(text)
                results.append(ai_diagnostic.test_vision(vision))
            if kind in ("text", "all"):
                if text not in installed:
                    raise ValueError(f"Install the selected text model: {text}")
                local_ai.release_model(vision)
                results.append(ai_diagnostic.test_text(text))
            return "\n\n".join(results)

        self.execute(work, self.results.setPlainText)
