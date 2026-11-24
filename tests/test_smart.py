from dataclasses import replace
from fractions import Fraction
import unittest

from movie_review_studio.model import Clip, Signature, timeline_entries, validate
from movie_review_studio.smart import AudioEvidence, assemble, caption_ranges


class SmartEditingTests(unittest.TestCase):
    def test_automatic_generation_without_highlights_and_with_chronology(self):
        fps = Fraction(24000, 1001)
        candidates = []
        for index in range(310):
            value = index * 97 % 256
            signature = Signature(index * 0x45D9F3B, f'{value:02x}' * 256)
            start = index * 120
            candidates.append(Clip(start, start + 119, float(index % 7), (signature,),
                                   (signature,), (signature,), shot=index))
        candidates[0] = replace(candidates[0], production_card=True, score=999)
        clips = assemble(candidates, [], fps, 39000, 1050, AudioEvidence(), lambda _: None)
        validate(clips, fps, 39000, final=True, has_audio=False)
        self.assertEqual(clips, sorted(clips, key=lambda clip: clip.start))
        self.assertNotIn(candidates[0], clips)
        entries = timeline_entries(clips, fps)
        self.assertTrue(all(entry.segment.frames / fps <= 5 for entry in entries))
        self.assertLessEqual(abs(sum(clip.frames for clip in clips) / fps - 1050), 5)
        self.assertTrue(all(a.position + a.segment.frames == b.position for a, b in zip(entries, entries[1:])))

    def test_subtitle_timestamps_supply_language_independent_dialogue_evidence(self):
        captions = caption_ranges('00:01:02,500 --> 00:01:04,000\n字幕\n\n01:05.000 --> 01:07.500\nOtro diálogo')
        self.assertEqual(captions, ((62.5, 64.0), (65.0, 67.5)))
        evidence = AudioEvidence(captions=captions)
        self.assertGreater(evidence.score(63, 64), evidence.score(4, 5))
