from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Rect = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass(frozen=True)
class LevelSpec:
    """Complete, serializable geometry for one generated level."""

    seed: int
    difficulty: int
    fire_start: Point
    water_start: Point
    fire_goal: Point
    water_goal: Point
    lava_rect: Rect
    water_rect: Rect
    platforms: tuple[Rect, ...]

    def as_env_config(self) -> dict[str, Point | Rect | list[Rect]]:
        return {
            "fire_start": self.fire_start,
            "water_start": self.water_start,
            "fire_goal": self.fire_goal,
            "water_goal": self.water_goal,
            "lava_rect": self.lava_rect,
            "water_rect": self.water_rect,
            "platforms": list(self.platforms),
        }


def _door_platform(
    goal_x: float,
    goal_y: float,
    width: float,
    screen_width: float,
) -> Rect:
    half_width = width / 2.0
    x1 = max(0.0, goal_x - half_width)
    x2 = min(screen_width, goal_x + half_width)
    return (x1, goal_y, x2, goal_y + 10.0)


def generate_level(
    seed: int,
    *,
    width: float = 800.0,
    height: float = 400.0,
    ground_y: float = 320.0,
    difficulty: int | None = None,
) -> LevelSpec:
    """Generate a deterministic, physically reachable full-screen layout.

    The generator varies spawn points, pool widths/gaps, goal positions,
    raised-door heights, and approach ledges. Every layout preserves the
    cooperative left-to-right/right-to-left objective and remains within the
    movement envelope of the environment's jump physics.
    """

    seed = int(seed)
    rng = np.random.default_rng(seed)
    if difficulty is None:
        difficulty = seed % 3
    if difficulty not in (0, 1, 2):
        raise ValueError("difficulty must be 0, 1, or 2")

    fire_start_x = float(rng.uniform(70.0, 135.0))
    water_start_x = float(rng.uniform(width - 135.0, width - 70.0))
    water_goal_x = float(rng.uniform(95.0, 180.0))
    fire_goal_x = float(rng.uniform(width - 180.0, width - 95.0))

    gap = float(rng.uniform(14.0, 42.0))
    lava_width = float(rng.uniform(45.0, 78.0))
    water_width = float(rng.uniform(45.0, 78.0))
    center = float(rng.uniform(width * 0.47, width * 0.53))
    lava_x2 = center - gap / 2.0
    lava_x1 = lava_x2 - lava_width
    water_x1 = center + gap / 2.0
    water_x2 = water_x1 + water_width

    lava_rect = (lava_x1, ground_y, lava_x2, height)
    water_rect = (water_x1, ground_y, water_x2, height)

    if difficulty == 0:
        fire_height = 0.0
        water_height = 0.0
    elif difficulty == 1:
        fire_height = float(rng.uniform(45.0, 82.0))
        water_height = float(rng.uniform(45.0, 82.0))
    else:
        # A jump peaks about 125 px above ground. Keeping platforms below
        # 108 px leaves enough vertical tolerance for reliable landings.
        fire_height = float(rng.uniform(76.0, 108.0))
        water_height = float(rng.uniform(76.0, 108.0))

    fire_goal = (fire_goal_x, ground_y - fire_height)
    water_goal = (water_goal_x, ground_y - water_height)
    platforms: list[Rect] = []

    if fire_height:
        platforms.append(
            _door_platform(
                fire_goal_x,
                fire_goal[1],
                float(rng.uniform(92.0, 132.0)),
                width,
            )
        )
    if water_height:
        platforms.append(
            _door_platform(
                water_goal_x,
                water_goal[1],
                float(rng.uniform(92.0, 132.0)),
                width,
            )
        )

    # Upper ledges add layout variation without creating hidden ceiling traps
    # in the direct route. Later generator families can make these interactive
    # once their topology is represented explicitly to the planner.
    if difficulty >= 1:
        left_center = float(rng.uniform(width * 0.25, width * 0.32))
        right_center = float(rng.uniform(width * 0.68, width * 0.75))
        ledge_width = float(rng.uniform(45.0, 75.0))
        ledge_y = float(rng.uniform(105.0, 155.0))
        platforms.extend(
            [
                (
                    left_center - ledge_width / 2.0,
                    ledge_y,
                    left_center + ledge_width / 2.0,
                    ledge_y + 10.0,
                ),
                (
                    right_center - ledge_width / 2.0,
                    ledge_y,
                    right_center + ledge_width / 2.0,
                    ledge_y + 10.0,
                ),
            ]
        )

    if difficulty == 2:
        # One high central shelf forces the policy to distinguish layouts
        # without obstructing the lower pool-crossing arc.
        shelf_center = float(rng.uniform(width * 0.47, width * 0.53))
        shelf_width = float(rng.uniform(48.0, 76.0))
        shelf_y = float(rng.uniform(115.0, 155.0))
        platforms.append(
            (
                shelf_center - shelf_width / 2.0,
                shelf_y,
                shelf_center + shelf_width / 2.0,
                shelf_y + 10.0,
            )
        )

    if len(platforms) > 8:  # pragma: no cover - protects future extensions
        raise RuntimeError("generated layouts may contain at most 8 platforms")

    return LevelSpec(
        seed=seed,
        difficulty=difficulty,
        fire_start=(fire_start_x, ground_y),
        water_start=(water_start_x, ground_y),
        fire_goal=fire_goal,
        water_goal=water_goal,
        lava_rect=lava_rect,
        water_rect=water_rect,
        platforms=tuple(platforms),
    )


def training_seeds(count: int, *, start: int = 0) -> list[int]:
    if count < 1:
        raise ValueError("count must be at least 1")
    return list(range(start, start + count))


def held_out_seeds(count: int, *, start: int = 100_000) -> list[int]:
    """Return seeds deliberately disjoint from the default training range."""
    if count < 1:
        raise ValueError("count must be at least 1")
    return list(range(start, start + count))
