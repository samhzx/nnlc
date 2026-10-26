"""Input feature contracts shared by the NNLC trainer and extractor."""

from __future__ import annotations

INPUT_DIMENSIONS = (18, 23)
DEFAULT_INPUT_DIM = 18
FUTURE_OFFSETS = (0.3, 0.6, 1.0, 1.5)
PAST_OFFSETS = {
    18: (-0.3, -0.2, -0.1),
    23: (-2.0, -1.0, -0.3, -0.2, -0.1),
}


def temporal_offsets(input_dim: int = DEFAULT_INPUT_DIM) -> tuple[float, ...]:
    if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim not in PAST_OFFSETS:
        raise ValueError(f"input_dim must be one of {INPUT_DIMENSIONS}")
    return (*PAST_OFFSETS[input_dim], *FUTURE_OFFSETS)


def temporal_suffix(offset: float) -> str:
    return f"_t{offset:+.1f}".replace(".", "").replace("+", "p").replace("-", "m")


def input_variables(input_dim: int = DEFAULT_INPUT_DIM) -> tuple[str, ...]:
    offsets = temporal_offsets(input_dim)
    current = ["v_ego", "desired_lateral_accel", "friction_input", "roll"]
    if input_dim == 23:
        current.append("a_ego")
    return tuple(
        current
        + ["desired_lateral_accel" + temporal_suffix(offset) for offset in offsets]
        + ["roll" + temporal_suffix(offset) for offset in offsets]
    )


def dimension_suffix(input_dim: int = DEFAULT_INPUT_DIM) -> str:
    temporal_offsets(input_dim)
    return f"{input_dim}d"


INPUT_PROFILES = {
    input_dim: {
        "input_dim": input_dim,
        "suffix": dimension_suffix(input_dim),
        "offsets": temporal_offsets(input_dim),
        "input_vars": input_variables(input_dim),
    }
    for input_dim in INPUT_DIMENSIONS
}
