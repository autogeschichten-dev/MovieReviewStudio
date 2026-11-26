"""JSON boundary types for exact timing used in cache keys and diagnostics."""
from fractions import Fraction


def exact(value):
    """Represent rational values losslessly without changing native JSON values."""
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, dict):
        return {key: exact(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [exact(item) for item in value]
    return value


def duration_fields(value):
    # Video Recap Skills expects numeric seconds. Preserve an exact companion for project reload.
    if isinstance(value, Fraction):
        return {"duration": float(value), "duration_ratio": exact(value)}
    return {"duration": value}
