from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from movie_review_studio import adaptive, narration
from movie_review_studio.media import ffmpeg, inspect
from movie_review_studio.model import Clip, timeline_entries
from movie_review_studio.render import RATE, Mix, audio_spans, render


def clip(start, frames, role=""):
    return Clip(start, start + frames, 1, (), (), (), highlight=bool(role), shot=start, role=role)


def digest(path):
    with wave.open(str(path), "rb") as source:
        result = sha256()
        while chunk := source.readframes(RATE * 4):
            result.update(chunk)
        return result.hexdigest()


class NarrationFirstTests(unittest.TestCase):
    def test_no_safe_internal_pause_skips_only_intermediate_group(self):
        fps = Fraction(2)
        movie = SimpleNamespace(fps=fps, has_audio=True)
        analysis = SimpleNamespace(frames=500)
        opening, highlight, ending = clip(0, 20, "Opening"), clip(80, 24, "Highlight"), clip(210, 20, "Ending")
        left, right = clip(35, 8), clip(140, 8)
        notices = []
        recording = narration.Recording(Path("fixture"), 8 * RATE, ())
        with patch.object(adaptive, "quiet_ranges", return_value=[]):
            picture, _, schedule, length, _ = adaptive.fit_recording(recording, [opening, left, highlight, right, ending],
                movie, analysis, None, 5, lambda _: None, notices=notices)
        self.assertEqual(picture, [opening, left, right, ending])
        self.assertEqual((schedule, length), (((0, 0, 8 * RATE),), 8 * RATE))
        self.assertTrue(any("Skipped original audio" in line for line in notices))

    def test_two_intermediates_stay_in_order_between_bookends(self):
        movie = SimpleNamespace(fps=Fraction(1), has_audio=True)
        analysis = SimpleNamespace(frames=500)
        opening, ending = clip(0, 10, "Opening"), clip(400, 10, "Ending")
        first, second = clip(120, 12, "Highlight"), clip(240, 12, "Highlight")
        picture = [opening, clip(30, 5), first, clip(170, 5), second, clip(300, 5), ending]
        recording = narration.Recording(Path("fixture"), 15 * RATE, ())
        quiet = [(5 * RATE, 5 * RATE), (10 * RATE, 10 * RATE)]
        with patch.object(adaptive, "quiet_ranges", return_value=quiet):
            result, _, schedule, length, _ = adaptive.fit_recording(
                recording, picture, movie, analysis, None, 5, lambda _: None)
        self.assertEqual(result, picture)
        self.assertEqual([end for _, _, end in schedule], [5 * RATE, 10 * RATE, 15 * RATE])
        self.assertEqual(length, recording.frames)

    def test_grows_picture_from_grounded_cached_alternatives(self):
        from movie_review_studio.director import ScriptSection

        movie = SimpleNamespace(fps=Fraction(1), has_audio=False)
        first = clip(10, 4)
        alternate = clip(30, 4)
        unrelated = replace(clip(80, 4), score=20)
        analysis = SimpleNamespace(frames=100, candidates=[first, alternate, unrelated])
        recording = narration.Recording(Path("fixture"), 8 * RATE, ())
        script = [ScriptSection("part-1", "A house by the river.", ((10, 14, "House"),), ((30, 34),))]
        with patch.object(adaptive, "quiet_ranges", return_value=[]):
            picture, _, schedule, _, _ = adaptive.fit_recording(recording, [first], movie, analysis, None,
                                                                5, lambda _: None, script=script)
        self.assertEqual(picture, [first, alternate])
        self.assertEqual(schedule[-1][-1], recording.frames)

    def test_subtitles_veto_a_waveform_pause_inside_spoken_cue(self):
        quiet = ((3 * RATE, 4 * RATE), (5 * RATE, 6 * RATE))
        self.assertEqual(adaptive.voiced_cues(quiet, [{"start": 2, "end": 4.5}]), [(5 * RATE, 6 * RATE)])


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class NarrationFirstMediaTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.progress = lambda _: None

    def write_part(self, index, duration, rate=8000):
        path = self.root / f"part-{index}.wav"
        ffmpeg("-f", "lavfi", "-i", f"sine=f={390 + index * 80}:r={rate}:d={duration}", str(path))
        return narration.read_part(path, self.root, self.progress)

    def test_reported_duration_three_parts_find_verified_pause_outside_old_window(self):
        parts = [self.write_part(index, duration) for index, duration in enumerate((200, 260, 362.633), 1)]
        merged = narration.merge(parts, 250, False, self.root, self.progress)
        self.assertAlmostEqual(merged.duration, 823.133, places=3)
        self.assertEqual(narration.merge(parts, 250, False, self.root, self.progress), merged)
        fps = Fraction(24000, 1001)
        movie = SimpleNamespace(fps=fps, has_audio=True)
        analysis = SimpleNamespace(frames=90000)
        opening, intermediate, ending = clip(0, 240 - 1, "Opening"), clip(32000, 288, "Highlight"), clip(80000, 239, "Ending")
        total = adaptive.narration_frames(merged, fps)
        left = [clip(1000 + 245 * i, 119) for i in range(80)]
        before = sum(item.frames for item in left)
        remainder = total + 19 - before
        full, tail = divmod(remainder, 119)
        right = [clip(33000 + 245 * i, 119) for i in range(full)]
        if tail:
            right.append(clip(33000 + 245 * full, tail))
        edit = [opening, *left, intermediate, *right, ending]
        # V3.3.1 searched only inside the interval made possible by its two fixed runs.
        old_low = round((before - 19) * RATE / fps)
        old_high = round(before * RATE / fps)
        self.assertIsNone(narration.last_pause(narration.quiet_ranges(merged), old_low, old_high))
        notices = []
        with patch.object(adaptive, "resized",
                          side_effect=lambda _, __, item, a, b, cache: replace(item, start=a, end=b)):
            picture, _, schedule, length, _ = adaptive.fit_recording(
                merged, edit, movie, analysis, self.root, 5, self.progress, notices=notices)
        self.assertIn(intermediate, picture)
        self.assertTrue(any("Moved original-audio cut" in note for note in notices))
        self.assertEqual(sum(c.frames for c in picture if not c.movie_audio), total)
        self.assertTrue(all(e.segment.frames / fps <= 5 for e in timeline_entries(picture, fps)))
        first_pause = schedule[0][2]
        second_part_boundary = merged.boundaries[2][0]
        self.assertLess(abs(first_pause - second_part_boundary), RATE)
        self.assertEqual(sum(end - start for _, start, end in schedule), merged.frames)
        with wave.open(str(narration.materialize(merged, schedule, length, self.root)), "rb") as bed, \
             wave.open(str(merged.path), "rb") as dry:
            remaining = merged.frames
            while remaining:
                count = min(remaining, RATE * 5)
                self.assertEqual(bed.readframes(count), dry.readframes(count))
                remaining -= count

    def test_repositioned_highlight_and_bookends_render_to_mp4(self):
        movie_path = self.root / "film.mp4"
        ffmpeg("-f", "lavfi", "-i", "testsrc2=s=96x64:r=4:d=35", "-f", "lavfi", "-i", "sine=f=300:d=35",
               "-c:v", "libx264", "-c:a", "aac", str(movie_path))
        parts = [self.write_part(i, 1.5, rate=48000) for i in (1, 2, 3)]
        merged = narration.merge(parts, 500, False, self.root, self.progress)
        movie = inspect(movie_path)
        analysis = SimpleNamespace(offset=0, frames=140, shot_starts=())
        opening, highlight, ending = clip(0, 8, "Opening"), clip(58, 8, "Highlight"), clip(126, 8, "Ending")
        first, last = clip(32, 8), clip(90, 14)
        picture, _, schedule, length, _ = adaptive.fit_recording(
            merged, [opening, first, highlight, last, ending], movie, analysis, self.root, 5, self.progress)
        self.assertEqual([c for c in picture if c.movie_audio], [opening, highlight, ending])
        self.assertEqual([original for original, _, _ in audio_spans(picture)], [True, False, True, False, True])
        bed = narration.materialize(merged, schedule, length, self.root)
        self.assertEqual(digest(bed), digest(merged.path))
        output = render(movie, analysis, picture, bed, None, self.root, self.progress, mix=Mix(),
                        destination=self.root / "review.mp4", managed_voice=True, fitted_duration=True)
        inspected = inspect(output)
        self.assertTrue(inspected.has_audio)
        self.assertAlmostEqual(inspected.duration, 11.5)
