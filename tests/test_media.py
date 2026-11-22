from pathlib import Path
from fractions import Fraction
from dataclasses import replace
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from movie_review_studio.analysis import analyze, fingerprint_highlight, thumbnail_path
from movie_review_studio.media import ffmpeg, frame_clock, identity, inspect
from movie_review_studio.model import Clip, timeline_entries
from movie_review_studio.render import RATE, audio_track, dimensions, render, render_clip, render_selection


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Requires FFmpeg and ffprobe")
class MediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.movie = cls.root / "movie with spaces.mp4"
        inputs = []
        for color in ("red", "green", "blue", "white"):
            inputs += ["-f", "lavfi", "-i", f"color={color}:s=96x64:r=12:d=5"]
        ffmpeg(*inputs, "-f", "lavfi", "-i", "sine=f=880:d=20",
               "-filter_complex", "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0[v]",
               "-map", "[v]", "-map", "4:a", "-c:v", "libx264", "-c:a", "aac", str(cls.movie))
        cls.media = inspect(cls.movie)

    def test_frame_accurate_cut_at_shot_boundary(self):
        for start, channel in [(59, 0), (60, 1), (119, 1), (120, 2)]:
            with self.subTest(start=start):
                path = self.root / "cut.mp4"
                render_clip(self.media, Clip(start, start + 1, 0, (), (), ()), path, True)
                pixels = ffmpeg("-i", str(path), "-vf", "scale=1:1", "-pix_fmt", "rgb24",
                                "-f", "rawvideo", "-threads", "1", "pipe:1", binary=True)
                self.assertEqual(len(pixels), 3)
                self.assertEqual(np.argmax(list(pixels)), channel)
        self.assertEqual(dimensions(self.media, False), (1620, 1080))

    def test_analysis_decodes_entire_movie_and_uses_cache(self):
        movie = self.root / "pattern.mp4"
        ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x90:rate=12:duration=3",
               "-f", "lavfi", "-i", "smptebars=size=160x90:rate=12:duration=3",
               "-f", "lavfi", "-i", "color=black:size=160x90:rate=12:duration=2",
               "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0", "-c:v", "libx264", str(movie))
        messages = []
        media = inspect(movie)
        analysis = analyze(media, self.root, messages.append)
        self.assertEqual(analysis.frames, 96)
        self.assertTrue(any(clip.end == 36 for clip in analysis.candidates))
        self.assertTrue(any(clip.start == 36 for clip in analysis.candidates))
        self.assertTrue(all(clip.end <= 72 for clip in analysis.candidates))
        self.assertTrue(all(thumbnail_path(self.root, identity(movie), clip.start).is_file()
                            for clip in analysis.candidates))
        self.assertEqual(analysis, analyze(media, self.root, messages.append))
        self.assertIn("Loaded cached shot analysis.", messages)

    def test_assembled_preview_audio_routes_and_cache(self):
        analysis = analyze(self.media, self.root, lambda _: None)
        highlight = fingerprint_highlight(self.media, analysis, 24, 144, cache=self.root)
        clips = [Clip(0, 12, 0, (), (), ()), highlight, Clip(150, 162, 0, (), (), ())]
        entries = timeline_entries(clips, self.media.fps)
        self.assertEqual([(e.position, e.segment.start, e.segment.end) for e in entries],
                         [(0, 0, 12), (12, 24, 60), (48, 60, 120), (108, 120, 144), (132, 150, 162)])
        self.assertTrue(all(e.segment.frames / self.media.fps <= 5 for e in entries))
        self.assertTrue(all(thumbnail_path(self.root, identity(self.movie), e.segment.start).is_file()
                            for e in entries if e.segment.highlight))
        voice = self.root / "voice.wav"
        music = self.root / "music.wav"
        ffmpeg("-f", "lavfi", "-i", "aevalsrc=if(lt(t\\,1)\\,0.1*sin(2*PI*440*t)\\,0.1*sin(2*PI*660*t)):d=2",
               str(voice))
        ffmpeg("-f", "lavfi", "-i", "sine=f=220:d=0.7", str(music))
        mix = self.root / "mix"
        mix.mkdir()
        output = audio_track(self.media, clips, voice, music, mix)
        with wave.open(str(output), "rb") as result:
            self.assertEqual(result.getnframes(), 12 * RATE)
            audio = np.frombuffer(result.readframes(12 * RATE), dtype="<i2")[::2].astype(float)

        def energy(start, frequency):
            data = audio[int(start * RATE):int((start + .1) * RATE)]
            return abs(np.fft.rfft(data)[frequency // 10]) / len(data)

        self.assertGreater(energy(.4, 440), 300)
        self.assertGreater(energy(.4, 220), 80)
        self.assertLess(energy(.4, 880), 3)
        self.assertGreater(energy(5, 880), 300)
        self.assertGreater(energy(3.98, 880), 300)  # Audio continues across an internal video cut at 4 s.
        self.assertLess(abs(audio[RATE + 1]), 100)
        self.assertLess(abs(audio[11 * RATE - 1]), 100)
        self.assertLess(energy(5, 440), 3)
        self.assertLess(energy(5, 220), 3)
        self.assertGreater(energy(11.4, 660), 300)
        self.assertGreater(energy(11.4, 220), 80)
        self.assertGreater(energy(.4, 220), energy(.9, 220) * 2)
        self.assertGreater(energy(11.4, 220), energy(11, 220) * 2)
        path = render(self.media, analysis, clips, voice, music, self.root, lambda _: None)
        self.assertAlmostEqual(inspect(path).duration, 12.0, places=2)
        self.assertEqual(frame_clock(inspect(path))[0], 144)
        self.assertEqual(path, render(self.media, analysis, clips, voice, music, self.root,
                                     lambda _: self.fail("Expected cached preview")))

        def pixels(video):
            raw = ffmpeg("-i", str(video), "-vf", "scale=1:1", "-pix_fmt", "rgb24",
                         "-f", "rawvideo", "-threads", "1", "pipe:1", binary=True)
            return np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)

        assembly_pixels = pixels(path)
        self.assertTrue(np.all(assembly_pixels.max(axis=1) > 80))
        np.testing.assert_array_equal(assembly_pixels.argmax(axis=1), [0] * 48 + [1] * 60 + [2] * 36)
        selected = render_selection(self.media, analysis, clips, 2, voice, music, self.root, lambda _: None)
        np.testing.assert_array_equal(pixels(selected), assembly_pixels[48:108])
        selected_audio = ffmpeg("-i", str(selected), "-map", "0:a:0", "-ac", "1", "-f", "s16le",
                                "pipe:1", binary=True)
        signal = np.frombuffer(selected_audio, dtype="<i2")[:RATE].astype(float)
        self.assertEqual(np.argmax(abs(np.fft.rfft(signal))), 880)

    def test_fractional_fps_cuts(self):
        path = self.root / "fractional.mkv"
        ffmpeg("-f", "lavfi", "-i", "testsrc2=size=96x64:rate=24000/1001:duration=3",
               "-c:v", "libx264", str(path))
        media = inspect(path)
        self.assertEqual(media.fps, Fraction(24000, 1001))
        analysis = analyze(media, self.root, lambda _: None)
        self.assertEqual(analysis.frames, 72)
        render_clip(media, Clip(23, 72, 0, (), (), ()), self.root / "fractional-cut.mp4", True)
        clips = [Clip(29, 72, 0, (), (), ()), Clip(0, 29, 0, (), (), ())]
        output = render(media, analysis, clips, None, None, self.root, lambda _: None)
        self.assertEqual(inspect(output).fps, media.fps)
        self.assertEqual(frame_clock(inspect(output))[0], 72)

    def test_editor_scene_controls_and_project_round_trip(self):
        from PySide6.QtWidgets import QApplication, QMessageBox
        from movie_review_studio.app import Studio

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        application = QApplication.instance() or QApplication([])
        studio = Studio(self.root)
        self.addCleanup(studio.close)
        source = analyze(self.media, self.root, lambda _: None)
        candidates = [Clip(start, start + 12, 0, (), (), (), shot=index)
                      for index, start in enumerate((0, 42, 150, 170, 190))]
        analysis = replace(source, candidates=candidates)
        highlight = fingerprint_highlight(self.media, analysis, 24, 144, cache=self.root)
        studio.movie, studio.source_key, studio.analysis = self.media, identity(self.movie), analysis
        studio.highlights = [highlight]
        studio.set_clips([candidates[0], highlight, candidates[2]])

        self.assertEqual(studio.scenes.item(0, 4).text(), "Selected")
        self.assertEqual(studio.scenes.item(1, 4).text(), "Highlight")
        self.assertEqual(studio.scenes.item(3, 4).text(), "Unused")
        studio.scene_search.setText("unused")
        self.assertTrue(studio.scenes.isRowHidden(0))
        self.assertFalse(studio.scenes.isRowHidden(3))
        studio.table.selectRow(4)
        studio.scenes.selectRow(3)
        studio.replace_with_candidate()
        studio.move_clip(-1)
        studio.scenes.selectRow(4)
        studio.add_candidate()
        studio.delete_clip()
        self.assertEqual(studio.clips, [candidates[0], candidates[3], highlight])
        self.assertEqual(studio.table.rowCount(), 5)
        self.assertEqual([studio.table.item(2, col).text() for col in (1, 2, 3, 4, 6)],
                         ["00:00:02.000", "00:00:05.000", "00:00:02.000", "00:00:05.000", "ORIGINAL"])
        studio.table.selectRow(1)
        studio.montage_playback = True
        studio.position_changed(5250)
        self.assertEqual(studio.active_row, 3)
        self.assertEqual(studio.table.currentRow(), 1)

        audio = self.root / "project-audio.wav"
        ffmpeg("-f", "lavfi", "-i", "sine=f=220:d=1", str(audio))
        studio.voice = studio.music = audio
        studio.title.setText("Edited review")
        studio.year.setValue(2026)
        studio.target.setValue(18)
        project = self.root / "test-project.mrs.json"
        with patch("movie_review_studio.app.QFileDialog.getSaveFileName", return_value=(str(project), "")):
            studio.save_project()
        expected = studio.clips[:]
        self.assertEqual(json.loads(project.read_text(encoding="utf-8"))["highlights"], [[24, 144, [60, 120]]])

        def run_job(_message, operation, complete):
            complete(operation(lambda _: None))

        with patch("movie_review_studio.app.analyze", return_value=analysis), \
             patch("movie_review_studio.app.QFileDialog.getOpenFileName", return_value=(str(project), "")), \
             patch("movie_review_studio.app.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes), \
             patch.object(studio, "background", side_effect=run_job):
            studio.set_clips([])
            studio.open_project()
            self.assertEqual(studio.clips, expected)
            legacy = json.loads(project.read_text(encoding="utf-8"))
            legacy.update(version=1, highlights=[[24, 144]])
            project.write_text(json.dumps(legacy), encoding="utf-8")
            studio.open_project()
        self.assertEqual(studio.clips, expected)
        self.assertEqual([entry.segment.highlight for entry in studio.entries], [False, False, True, True, True])
        self.assertEqual((studio.voice, studio.music, studio.title.text(), studio.year.value(), studio.target.value()),
                         (audio, audio, "Edited review", 2026, 18))
        application.processEvents()
