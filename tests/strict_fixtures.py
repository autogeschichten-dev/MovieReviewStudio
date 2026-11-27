from dataclasses import replace

from movie_review_studio.model import Clip


def audio_group(group, spans):
    return replace(group, cuts=(), picture=tuple(
        Clip(start, end, 1, (), (), (), shot=start) for start, end in spans))
