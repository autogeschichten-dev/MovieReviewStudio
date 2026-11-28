"""Real cached analysis, Fast Montage, synchronization, save/reload and FFmpeg export."""
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
from unittest.mock import patch
import wave

import numpy as np

from movie_review_studio.media import ffmpeg, frame_clock, identity, inspect
from movie_review_studio.model import timeline_entries
from movie_review_studio.render import RATE


@skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class MandatoryAudioPipelineTests(TestCase):
    def test_default_original_audio_survives_generation_export_and_reload(self):
        from PySide6.QtWidgets import QApplication, QMessageBox
        from movie_review_studio.app import Studio

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        application = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "movie.mp4"
            self.make_movie(root, source)
            studio = Studio(root)
            self.addCleanup(studio.close)
            media = inspect(source)
            studio.apply_movie((media, identity(source)))
            voices = []
            for i, duration in enumerate((3.25, 2.5, 2.5, 2.5)):
                path = root / f"narration-{i}.wav"
                ffmpeg("-f", "lavfi", "-i", f"sine=f=1100:r={RATE}:d={duration}", str(path))
                voices.append(path)
            music = root / "music.wav"
            ffmpeg("-f", "lavfi", "-i", f"sine=f=170:r={RATE}:d=2", str(music))
            studio.music = music
            studio.voice_pause = 1500
            studio.voice_trim = False

            def inline(_message, work, apply, **_kwargs):
                apply(work(lambda _: None))

            with patch.object(studio, "background", side_effect=inline), \
                 patch.object(studio, "error", side_effect=lambda message: self.fail(message)), \
                 patch.object(studio, "approve_audio_omissions", side_effect=self.no_omissions), \
                 patch("movie_review_studio.app.QMessageBox.information"):
                studio.load_voice_parts(0, voices)
                studio.generate()
                self.assertTrue(studio.voice_sync_valid())
                original = [clip for clip in studio.clips if clip.movie_audio]
                self.assertEqual([group.role for group in original], ["Opening", "Highlight", "Highlight", "Highlight", "Ending"])
                self.assertEqual([group.frames for group in original], [10, 12, 12, 12, 10])
                self.assertEqual(studio.voice_recording.duration, 15.25)
                self.assertEqual(studio.validation_report()["violations"], [])
                self.assertEqual(studio.validation_report()["original_audio_visual_exceptions"], 0)
                with wave.open(str(studio.voice), "rb") as bed, wave.open(str(studio.voice_recording.path), "rb") as dry:
                    self.assertEqual(bed.readframes(dry.getnframes()), dry.readframes(dry.getnframes()))
                self.export_and_verify(studio, root / "first.mp4")
                with patch.object(studio, "load_player") as load:
                    studio.preview(studio.clips, assembly=True)
                    self.verify_file(studio, load.call_args.args[0])

                project = root / "review.mrs.json"
                with patch("movie_review_studio.app.QFileDialog.getSaveFileName", return_value=(str(project), "")):
                    studio.save_project()
                expected = studio.clips[:]
                with patch("movie_review_studio.app.QFileDialog.getOpenFileName", return_value=(str(project), "")), \
                     patch("movie_review_studio.app.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
                    studio.open_project()
                self.assertEqual(studio.clips, expected)
                self.assertTrue(studio.voice_sync_valid())
                self.export_and_verify(studio, root / "reopened.mp4")
            studio.close()
        application.processEvents()

    def no_omissions(self, omissions):
        self.assertFalse(omissions)
        return True

    @staticmethod
    def make_movie(root, output, shots=106):
        width, height = 128, 72
        bits = max(7, (shots - 1).bit_length())
        stride = width // bits if bits > 7 else 12
        raw = root / "picture.bgr"
        rng = np.random.default_rng(343)
        with raw.open("wb") as stream:
            for index in range(shots):
                tiles = rng.integers(30, 224, (9, 16, 3), dtype=np.uint8)
                image = np.repeat(np.repeat(tiles, 8, axis=0), 8, axis=1)
                for bit in range(bits):
                    image[0:8, bit * stride:(bit + 1) * stride] = 238 if (index >> bit) & 1 else 16
                stream.write(image.tobytes() * 4)
        # The original sound is distinct from narration (1100 Hz) and music (170 Hz).
        signal = rf"aevalsrc=0.20*sin(2*PI*(330+37*mod(floor(t/4)\,7))*t):s={RATE}:d={shots * 4}"
        ffmpeg("-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{width}x{height}",
               "-framerate", "1", "-i", str(raw), "-f", "lavfi", "-i", signal,
               "-c:v", "libx264", "-threads", "2", "-preset", "ultrafast", "-c:a", "aac", str(output))

    def export_and_verify(self, studio, path):
        with patch("movie_review_studio.app.QFileDialog.getSaveFileName", return_value=(str(path), "")):
            studio.export()
        self.verify_file(studio, path)

    def verify_file(self, studio, path):
        info = inspect(path)
        self.assertTrue(info.has_audio)
        self.assertEqual(frame_clock(info)[0], 72)
        self.assertEqual(info.duration, 72)
        video = ffmpeg("-i", str(path), "-an", "-vf", "scale=128:72", "-pix_fmt", "rgb24",
                       "-f", "rawvideo", "-threads", "1", "pipe:1", binary=True)
        frames = np.frombuffer(video, dtype=np.uint8).reshape((-1, 72, 128, 3))
        ids = sum((frames[:, 4, bit * 12 + 6, 0] > 128).astype(int) << bit for bit in range(7))
        expected = [source // 4 for row in timeline_entries(studio.clips, info.fps)
                    for source in range(row.segment.start, row.segment.end)]
        np.testing.assert_array_equal(ids, expected)
        raw = ffmpeg("-i", str(path), "-map", "0:a:0", "-ar", str(RATE), "-ac", "1", "-f", "s16le",
                     "pipe:1", binary=True)
        pcm = np.frombuffer(raw, dtype="<i2")
        with wave.open(str(studio.voice_recording.path), "rb") as source:
            dry = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").reshape(-1, source.getnchannels())[:, 0]
        cursor = narrated = 0
        checked_speech_samples = 0
        for clip in studio.clips:
            if clip.movie_audio:
                for offset in range(clip.frames):
                    second = cursor + offset
                    signal = pcm[second * RATE + RATE // 4:second * RATE + 3 * RATE // 4]
                    spectrum = np.abs(np.fft.rfft(signal))
                    original = (330 + 37 * (((clip.start + offset) // 4) % 7)) / 2
                    peak = int(np.argmax(spectrum))
                    self.assertLessEqual(abs(peak - original), 1)
                    self.assertLess(spectrum[550], spectrum[peak] * .03)
                    self.assertLess(spectrum[85], spectrum[peak] * .30)
            else:
                for offset in range(0, clip.frames * RATE, RATE // 4):
                    reference = dry[narrated + offset:narrated + offset + RATE // 4].astype(float)
                    if len(reference) != RATE // 4:
                        continue
                    expected_level = np.abs(np.fft.rfft(reference))[275]
                    if expected_level < 100000:
                        continue
                    start = cursor * RATE + offset
                    spectrum = np.abs(np.fft.rfft(pcm[start:start + RATE // 4].astype(float)))
                    self.assertGreater(spectrum[275], expected_level * .60)
                    self.assertLess(max(spectrum[80:145]), spectrum[275] * .12)
                    checked_speech_samples += RATE // 4
                narrated += clip.frames * RATE
            cursor += clip.frames
        self.assertGreaterEqual(checked_speech_samples, 10 * RATE)
        self.assertGreater(np.max(np.abs(pcm)), 1000)
