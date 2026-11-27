from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import json

from strict_fixtures import audio_group

from movie_review_studio import local_ai, planning
from movie_review_studio.director import chapters, sample_group
from movie_review_studio.media import Cancelled, identity
from movie_review_studio.model import Clip, source_gap


class RecoveryTests(TestCase):
    def test_reference_errors_are_partial_and_not_positional(self):
        valid, invalid = planning.references(["s100", 100, {"id": "s120"}, {"id": 120},
                                              "0", "s200", {}], {"s100", "s120"})
        self.assertEqual(valid, ["s100", "s120"])
        self.assertEqual(len(invalid), 5)
        self.assertEqual(planning.references("s100", {"s100"})[0], [])

    def test_reject_unsupported_movie_identification_and_coordinates(self):
        observed = planning.visual_text(["A scene from the movie The Wizard of Oz. The image shows a boy beside a blue door.",
                                         "(30, 45), (50, 21)", "A boy named Adrian turns away."])
        self.assertEqual(observed, "a boy beside a blue door.")

    def test_context_truncation_is_reported_and_raw_reply_saved(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            planner = planning.TextPlanner(SimpleNamespace(model="qwen", url="fixture"), root / "report.jsonl")
            planner.local = True
            with patch.object(local_ai, "request") as call:
                def truncated(_route, _body, **kwargs):
                    kwargs["trace"]({"http_status": 200, "body": "fixture raw"})
                    return {"done_reason": "length", "message": {"content": '{"shots":['}}
                call.side_effect = truncated
                with self.assertRaisesRegex(ValueError, "output limit"):
                    planner.ask("chapter-1", {"task": "fixture"}, planning.shot_schema(["s0"]), root, lambda _: None)
                self.assertIn("fixture raw", planner.report.read_text())
                with self.assertRaisesRegex(ValueError, "planning context"):
                    planner.ask("chapter-1", {"task": "x" * 9999}, {}, root, lambda _: None)
                self.assertEqual(call.call_count, 1)

    def test_640_scenes_recover_and_resume_using_v311_visual_cache(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "film.mp4"
            source.write_bytes(b"fixture")
            movie = SimpleNamespace(path=source, fps=Fraction(1), width=96, height=64, has_audio=False)
            starts = [i * 8 - (4 if i and i % 8 == 0 else 0) for i in range(640)]
            clips = [Clip(start, start + 2, 1, (), (), (), shot=i) for i, start in enumerate(starts)]
            analysis = SimpleNamespace(frames=5120, candidates=clips)
            selected = [clip for group in chapters(clips, 16) for clip in sample_group(group)]
            self.assertEqual(len(selected), 640)
            batch = local_ai.VisionBatch(movie, analysis, root, "moondream", "existing-digest")
            from hashlib import sha256
            for clip in selected[5:]:
                stamp = f"{identity(source)}:moondream:existing-digest:{clip.start}:{clip.end}:v1"
                (root / f"ollama-{sha256(stamp.encode()).hexdigest()}.json").write_text(json.dumps({
                    "source_in": float(clip.start), "source_out": float(clip.end),
                    "timestamp": float(clip.start + 1), "description": "The image shows a family walks beside a wooden doorway. A young boy has short brown hair and a blue shirt standing in front of a wooden door."
                }), encoding="utf-8")
            with patch.object(local_ai, "describe_image") as vision:
                observations = batch.describe(selected, lambda _: None, cache_only=True)
                vision.assert_not_called()
            self.assertEqual((len(observations), batch.failed), (635, 5))

            provider = SimpleNamespace(model="mock-qwen", url=local_ai.OLLAMA + "/v1/chat/completions")
            report = root / "report.jsonl"
            stages = []
            selection_calls = 0
            interrupt = True
            summary_calls = 0
            repeat_options = []
            full_evidence = "short brown hair and a blue shirt standing in front of a wooden door."

            def request(route, body=None, **kwargs):
                nonlocal selection_calls, summary_calls
                if route == "/api/show":
                    return {"model_info": {"qwen.context_length": 2048}}
                if route == "/api/version":
                    return {"version": "test"}
                if route == "/api/tags":
                    return {"models": [{"name": "mock-qwen", "digest": "fixture"}]}
                schema = body["format"]
                field = schema["required"][0]
                if "shots" in schema["properties"]:
                    selection_calls += 1
                    if interrupt and selection_calls == 4:
                        raise Cancelled()
                    choices = schema["properties"]["shots"]["items"]["enum"]
                    stages.append(("chapter", choices))
                    selected_ids = [choices[0], "s999999", 0] if selection_calls <= 2 else choices
                    response = {"text": "The family walks beside the wooden doorway.", "shots": selected_ids}
                else:
                    stages.append((field, []))
                    if field == "summary":
                        summary_calls += 1
                        prompt = body["messages"][1]["content"]
                        self.assertIn(full_evidence, prompt)
                        if summary_calls in (15, 16):
                            repeat_options.append(body["options"])
                            return {"done_reason": "length", "prompt_eval_count": 697, "eval_count": 640,
                                    "message": {"content": '{"summary":"' +
                                                "a man in a blue shirt standing in front of a door " * 58}}
                    response = {field: "A family walks beside the wooden doorway."}
                return {"message": {"content": json.dumps(response)}, "done_reason": "stop"}

            def run():
                return planning.plan(movie, analysis, [], provider, "Cultural Stories", "English",
                    1050, "", root, lambda _: None, clips, observations, [], [], report)

            with patch.object(local_ai, "request", side_effect=request), \
                 patch.object(planning.recap_render, "validate_timeline_physical"):
                with self.assertRaises(Cancelled):
                    run()
                first_dir = next(root.glob("planning-v312-*"))
                first = json.loads((first_dir / "chapter-1.json").read_text())
                self.assertTrue(first["fallback"])
                self.assertEqual(len(repeat_options), 2)
                self.assertGreater(repeat_options[1]["repeat_penalty"], repeat_options[0]["repeat_penalty"])
                self.assertTrue((first_dir / "summary-15-1.json").is_file())
                self.assertTrue((first_dir / "summary-15-2.json").is_file())
                self.assertIn("repeated phrases", report.read_text(encoding="utf-8"))
                self.assertTrue(all(value.startswith("s") and value != "s999999" for value in first["shots"]))
                self.assertLess(len(stages[-2][1]), len(stages[-3][1]))
                interrupt = False
                stages.clear()
                timeline, sections = run()
                self.assertTrue(all(stage == "chapter" for stage, _ in stages))
                self.assertTrue(any("Fallback evidence match:" in obs for section in sections for _, _, obs in section.evidence))
                self.assertEqual(sum(clip.frames for clip in timeline), 1050)
                self.assertTrue(all(clip.frames <= 5 for clip in timeline))
                self.assertTrue(all(source_gap(a, b) >= 5 for a, b in zip(timeline, timeline[1:])))
                stale = first_dir / "chapter-1.json"
                stale.write_text('{"text":"stale","shots":["s999999"]}', encoding="utf-8")
                stages.clear()
                run()
                self.assertGreaterEqual(len(stages), 1)
                self.assertLessEqual(len(stages), 2)
                self.assertTrue((first_dir / "timeline.json").is_file())

    def test_duration_recovery_expands_cached_chapter_candidates_without_new_inference(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "film.mp4"
            source.write_bytes(b"fixture")
            movie = SimpleNamespace(path=source, fps=Fraction(1), width=96, height=64, has_audio=True)
            clips = [Clip(i * 12, i * 12 + 5, 1, (), (), (), shot=i) for i in range(300)]
            highlight = audio_group(Clip(3600, 3612, 1, (), (), (), highlight=True),
                                    [(3600, 3605), (3610, 3615), (3620, 3622)])
            analysis = SimpleNamespace(frames=3622, candidates=clips)

            def observation(clip, description):
                return {"timestamp": clip.start + 2, "source_in": clip.start, "source_out": clip.end,
                        "description": description}

            narration = "A family walks beside the wooden doorway."
            # A deliberate first-chapter shortage: most sampled clips have valid observations,
            # but those observations do not support the chapter narration.
            initial = {clip.start: observation(clip, "A tractor crosses an empty field." if index < 42 else narration)
                       for index, clip in enumerate(clips[:169]) if index not in range(42, 49)}
            calls = []
            messages = []

            def expand(batch, progress, *, cache_only):
                calls.append(cache_only)
                self.assertTrue(cache_only)
                return {clip.start: observation(clip, narration) for clip in batch if clip.start >= 169 * 12}

            def ask(self, stage, payload, schema, cache, progress, **kwargs):
                if "shots" in schema["properties"]:
                    choice = schema["properties"]["shots"]["items"]["enum"][0]
                    return {"shots": [choice], "text": narration}
                return {schema["required"][0]: narration}

            provider = SimpleNamespace(model="fixture", url="fixture")
            report = root / "duration.jsonl"

            def plan(expand_callback=expand, *, expand_live=False):
                return planning.plan(movie, analysis, [highlight], provider, "Cultural Stories", "English",
                                     1050, "", root, messages.append, clips, initial, [], [], report,
                                     expand=expand_callback, expand_live=expand_live)

            with patch.object(planning.TextPlanner, "ask", new=ask), \
                 patch.object(planning.recap_render, "validate_timeline_physical"):
                with self.assertRaisesRegex(ValueError, r"Maximum achievable duration with available evidence: [\d.]+ s"):
                    plan(None)
            with patch.object(planning.TextPlanner, "ask", side_effect=AssertionError("planning cache missed")), \
                 patch.object(planning.recap_render, "validate_timeline_physical"):
                timeline, sections = plan()
            self.assertEqual(calls, [True])
            self.assertEqual(sum(clip.frames for clip in timeline), 1052)
            self.assertIn(highlight, timeline)
            self.assertTrue(all(source_gap(a, b) >= 5 for a, b in zip(timeline, timeline[1:])))
            self.assertTrue(any(start >= 169 * 12 for section in sections for start, _, _ in section.evidence))
            self.assertTrue(any("no overlapping cues" in line for line in messages))
            self.assertTrue(any("eligible" in line and "rejected" in line for line in messages))

            calls.clear()
            def live(batch, progress, *, cache_only):
                calls.append(cache_only)
                if cache_only:
                    return {}
                return {clip.start: observation(clip, narration) for clip in batch if clip.start >= 169 * 12}

            with patch.object(planning.TextPlanner, "ask", side_effect=AssertionError("planning cache missed")), \
                 patch.object(planning.recap_render, "validate_timeline_physical"):
                timeline, _ = plan(live, expand_live=True)
            self.assertEqual(calls, [True, False, False])
            self.assertGreaterEqual(sum(clip.frames for clip in timeline), 1045)

    def test_capacity_uses_alternate_cut_and_rejects_repeated_shot(self):
        clips = [
            Clip(0, 2, 1, (), (), (), shot=0),
            Clip(3, 8, 1, (), (), (), shot=1),
            Clip(12, 17, 1, (), (), (), shot=1),
            Clip(22, 27, 1, (), (), (), shot=2),
        ]
        shots = {f"s{clip.start}": planning.Shot(f"s{clip.start}", clip, "wooden door", clip.start, "")
                 for clip in clips}
        route = planning.longest_route(shots, [], Fraction(1), 5)
        self.assertEqual([clip.start for clip, _ in route], [0, 12, 22])
        self.assertEqual(sum(clip.frames for clip, _ in route), 12)
