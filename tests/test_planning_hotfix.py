from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from movie_review_studio import adaptive, director, planning
from movie_review_studio.model import Clip, source_gap
from movie_review_studio.timing_json import exact


class FinalizationTests(unittest.TestCase):
    def test_exact_nested_timing_and_project_section_round_trip(self):
        ratio = Fraction(1001, 24000)
        raw = json.loads(json.dumps(exact({"rows": [{"timing": [ratio, 2, None]}]})))
        self.assertEqual(raw["rows"][0]["timing"][0], {"numerator": 1001, "denominator": 24000})
        section = director.ScriptSection("part-1", "Recorded narration", ((2, 10, "Location"),), duration=ratio)
        restored = director.ScriptSection.restore(json.loads(json.dumps(section.json_row())))
        self.assertEqual(restored, section)
        source = SimpleNamespace(fps=Fraction(24000, 1001), width=96, height=64, path=Path("film.mp4"))
        document = director.timeline_document(source, [Clip(2, 10, 0, (), (), ())], [section])
        self.assertEqual(document["canvas"]["fps"], "24000/1001")
        self.assertEqual(json.loads(json.dumps(document))["meta"]["script_sections"][0]["duration_ratio"],
                         exact(ratio))

    def test_fourteen_frames_reconciled_with_a_verified_unused_source_range(self):
        fps = Fraction(24000, 1001)
        movie = SimpleNamespace(fps=fps, has_audio=False)
        analysis = SimpleNamespace(frames=3200)
        first = Clip(10, 95, 1, (), (), (), shot=1)
        alternate = Clip(400, 504, 1, (), (), (), shot=2)
        conflict = Clip(101, 150, 1, (), (), (), shot=3)
        eligible = {str(c.start): planning.Shot(str(c.start), c, "A woman opens a door.", float((c.start + 25) / fps), "")
                    for c in (first, alternate, conflict)}
        owners = dict.fromkeys(eligible, 0)
        with TemporaryDirectory() as tmp, patch.object(
                adaptive, "resized", side_effect=lambda _, __, clip, start, end, cache: replace(clip, start=start, end=end)):
            root = Path(tmp)
            result = planning.reconcile_shortfall([(first, "10")], eligible, owners, movie, analysis, root,
                                                   99, 5, root / "report.jsonl")
            self.assertEqual([c.frames for c, _ in result], [85, 14])
            self.assertEqual(result[1][1], "400")
            self.assertGreaterEqual(source_gap(result[0][0], result[1][0]), fps * 5)
            self.assertEqual(json.loads((root / "report.jsonl").read_text())["exact_seconds"],
                             exact(Fraction(14) / fps))

    def test_thirty_chapters_resume_at_failed_finalization_with_rational_target(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "film.mp4"
            source.touch()
            fps = Fraction(24000, 1001)
            movie = SimpleNamespace(fps=fps, width=96, height=64, has_audio=False, path=source)
            clips = [Clip(i * 320, i * 320 + 100, 1, (), (), (), shot=i) for i in range(623)]
            analysis = SimpleNamespace(frames=200000)
            seen = {c.start: {"timestamp": float((c.start + 50) / fps), "source_in": float(c.start / fps),
                              "source_out": float(c.end / fps), "description": "A woman opens the wooden door."}
                    for c in clips}
            report = root / "report.jsonl"
            provider = SimpleNamespace(model="mock-qwen", url="fixture")
            goal = 24300
            target = goal / fps

            def response(_self, stage, payload, schema, cache, progress, **kwargs):
                field = schema["required"][0]
                result = {field: "A woman opens the wooden door."}
                if "shots" in schema["required"]:
                    result["shots"] = [key for key, _ in payload["candidates"]]
                return result

            def run():
                return planning.plan(movie, analysis, [], provider, "Cultural Stories", "English",
                                     target, "", root, lambda _: None, clips, seen, [], [], report, adaptive=True)

            real_write = planning.atomic_json
            def simulated_write(path, value):
                if path.name == "timeline.json":
                    raise TypeError("fixture final timeline persistence failure")
                real_write(path, value)

            with patch.object(planning.recap_render, "validate_timeline_physical"), \
                 patch.object(planning.TextPlanner, "ask", new=response), \
                 patch.object(planning, "atomic_json", side_effect=simulated_write):
                with self.assertRaisesRegex(TypeError, "fixture"):
                    run()
            directory = next(root.glob("planning-v312-*"))
            self.assertEqual(len(list(directory.glob("chapter-*.json"))), 30)
            self.assertTrue((directory / "finalization.json").exists())
            self.assertIn("Traceback (most recent call last)", report.read_text())
            with patch.object(planning.recap_render, "validate_timeline_physical"), \
                 patch.object(planning.TextPlanner, "ask", side_effect=AssertionError("must reuse completed planning")):
                picture, sections = run()
            self.assertEqual(sum(c.frames for c in picture), goal)
            self.assertEqual(len(sections), 30)
            self.assertFalse((directory / "finalization.json").exists())

    def test_visual_cleanup_preserves_completed_sentences(self):
        text = planning.visual_text(["A woman opens a door. She walks into", "x=20, y=50, w=4, h=2",
                                     "The child smiles."])
        self.assertEqual(text, "A woman opens a door. The child smiles.")
