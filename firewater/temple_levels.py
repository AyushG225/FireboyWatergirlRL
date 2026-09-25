from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Point = tuple[float, float]
Rect = tuple[float, float, float, float]


@dataclass(frozen=True)
class HazardSpec:
    kind: str
    rect: Rect


@dataclass(frozen=True)
class GemSpec:
    owner: str
    position: Point


@dataclass(frozen=True)
class ButtonSpec:
    button_id: str
    owner: str
    position: Point


@dataclass(frozen=True)
class GateSpec:
    gate_id: str
    rect: Rect
    required_buttons: tuple[str, ...]


@dataclass(frozen=True)
class MovingPlatformSpec:
    platform_id: str
    x1: float
    x2: float
    upper_y: float
    lower_y: float
    speed: float
    phase: float


@dataclass(frozen=True)
class TempleLevelSpec:
    """Complete geometry and puzzle state for a multi-floor temple."""

    seed: int
    fire_start: Point
    water_start: Point
    fire_goal: Point
    water_goal: Point
    platforms: tuple[Rect, ...]
    moving_platforms: tuple[MovingPlatformSpec, ...]
    hazards: tuple[HazardSpec, ...]
    gems: tuple[GemSpec, ...]
    buttons: tuple[ButtonSpec, ...]
    gates: tuple[GateSpec, ...]
    floor_y: tuple[float, ...]


def _jitter(rng: np.random.Generator, value: float, radius: float) -> float:
    return float(value + rng.uniform(-radius, radius))


def generate_temple_level(
    seed: int,
    *,
    width: float = 960.0,
    height: float = 720.0,
) -> TempleLevelSpec:
    """Generate a solvable, five-floor cooperative temple.

    The topology is deliberately stable: four alternating lifts form a
    guaranteed route from the bottom floor to the doors. Within that topology,
    the seed changes pool widths and positions, gem placement, door placement,
    and lift timing. This provides meaningful layout variation without
    producing impossible random geometry.
    """

    seed = int(seed)
    rng = np.random.default_rng(seed)
    if width < 800 or height < 600:
        raise ValueError("temple levels require at least an 800x600 world")

    scale_x = width / 960.0
    scale_y = height / 720.0

    def sx(value: float) -> float:
        return value * scale_x

    def sy(value: float) -> float:
        return value * scale_y

    floors = tuple(sy(value) for value in (660, 535, 410, 285, 160))
    left_lift = (sx(80), sx(155))
    right_lift = (sx(805), sx(880))

    # Ledges meet the lift shafts exactly. This avoids physics-dependent cracks
    # while the moving surface itself still has to be timed and boarded.
    platforms = (
        (sx(80), floors[1], sx(805), floors[1] + sy(22)),
        (sx(155), floors[2], sx(880), floors[2] + sy(22)),
        (sx(80), floors[3], sx(805), floors[3] + sy(22)),
        (sx(155), floors[4], sx(805), floors[4] + sy(24)),
        # Small upper alcoves make the silhouette closer to a temple room.
        (sx(30), sy(95), sx(230), sy(116)),
        (sx(730), sy(95), sx(930), sy(116)),
    )

    moving_platforms = (
        MovingPlatformSpec(
            "lift_0",
            right_lift[0],
            right_lift[1],
            floors[1],
            floors[0],
            sy(72.0),
            0.0,
        ),
        MovingPlatformSpec(
            "lift_1",
            left_lift[0],
            left_lift[1],
            floors[2],
            floors[1],
            sy(68.0),
            float(rng.uniform(0.0, 0.45)),
        ),
        MovingPlatformSpec(
            "lift_2",
            right_lift[0],
            right_lift[1],
            floors[3],
            floors[2],
            sy(70.0),
            float(rng.uniform(0.0, 0.45)),
        ),
        MovingPlatformSpec(
            "lift_3",
            left_lift[0],
            left_lift[1],
            floors[4],
            floors[3],
            sy(66.0),
            float(rng.uniform(0.0, 0.45)),
        ),
    )

    hazards: list[HazardSpec] = []

    def hazard(
        kind: str,
        center_x: float,
        floor_index: int,
        width_base: float,
    ) -> None:
        center = sx(_jitter(rng, center_x, 10.0))
        hazard_width = sx(_jitter(rng, width_base, 5.0))
        y = floors[floor_index]
        hazards.append(
            HazardSpec(
                kind,
                (
                    center - hazard_width / 2.0,
                    y - sy(3),
                    center + hazard_width / 2.0,
                    y + sy(18),
                ),
            )
        )

    # Each corridor contains complementary elemental pools plus one green pool
    # that is deadly to both. Their spacing stays within one-jump reach.
    for kind, x, floor_index, pool_width in (
        ("water", 310, 0, 58),
        ("lava", 500, 0, 62),
        ("acid", 675, 0, 52),
        ("lava", 650, 1, 58),
        ("acid", 465, 1, 48),
        ("water", 300, 1, 60),
        ("water", 315, 2, 58),
        ("lava", 505, 2, 58),
        ("acid", 675, 2, 48),
        ("lava", 650, 3, 56),
        ("water", 330, 3, 58),
    ):
        hazard(kind, x, floor_index, pool_width)

    buttons = (
        ButtonSpec("f_floor_1", "fire", (sx(800), floors[1])),
        ButtonSpec("w_floor_1", "water", (sx(780), floors[1])),
        ButtonSpec("f_floor_2", "fire", (sx(172), floors[2])),
        ButtonSpec("w_floor_2", "water", (sx(185), floors[2])),
        ButtonSpec("f_floor_3", "fire", (sx(800), floors[3])),
        ButtonSpec("w_floor_3", "water", (sx(780), floors[3])),
    )
    gates = (
        GateSpec(
            "gate_1",
            (sx(520), floors[2] + sy(22), sx(540), floors[1]),
            ("f_floor_1", "w_floor_1"),
        ),
        GateSpec(
            "gate_2",
            (sx(505), floors[3] + sy(22), sx(525), floors[2]),
            ("f_floor_2", "w_floor_2"),
        ),
        GateSpec(
            "gate_3",
            (sx(520), floors[4] + sy(24), sx(540), floors[3]),
            ("f_floor_3", "w_floor_3"),
        ),
    )

    # Gems sit close enough to the running line to be collected without
    # pixel-perfect jumps. Each player has one gem on three different floors.
    gems = (
        GemSpec("fire", (sx(_jitter(rng, 225, 8)), floors[0] - sy(30))),
        GemSpec("water", (sx(_jitter(rng, 405, 8)), floors[0] - sy(30))),
        GemSpec("fire", (sx(_jitter(rng, 610, 8)), floors[1] - sy(31))),
        GemSpec("water", (sx(_jitter(rng, 390, 8)), floors[1] - sy(31))),
        GemSpec("fire", (sx(_jitter(rng, 375, 8)), floors[2] - sy(31))),
        GemSpec("water", (sx(_jitter(rng, 620, 8)), floors[2] - sy(31))),
    )

    fire_goal_x = sx(_jitter(rng, 620, 18))
    water_goal_x = sx(_jitter(rng, 350, 18))
    return TempleLevelSpec(
        seed=seed,
        fire_start=(sx(105), floors[0]),
        water_start=(sx(165), floors[0]),
        fire_goal=(fire_goal_x, floors[4]),
        water_goal=(water_goal_x, floors[4]),
        platforms=platforms,
        moving_platforms=moving_platforms,
        hazards=tuple(hazards),
        gems=gems,
        buttons=buttons,
        gates=gates,
        floor_y=floors,
    )
