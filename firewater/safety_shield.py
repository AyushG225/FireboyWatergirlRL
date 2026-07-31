from __future__ import annotations

from firewater.firewater_env import FireWaterEnv


def shield_action(
    env: FireWaterEnv,
    proposed_action,
    *,
    margin: float = 70.0,
) -> int:
    """Override an action only when a grounded player is about to die.

    The shield is geometry-based and level-agnostic. If both players are at
    risk, it jumps the one with the shortest estimated time to its deadly
    pool; the other can be handled on the next frame.
    """
    if margin <= 0:
        raise ValueError("margin must be positive")

    risks: list[tuple[float, int]] = []

    def add_risk(
        *,
        x: float,
        vx: float,
        on_ground: bool,
        deadly_rect,
        jump_action: int,
    ):
        if not on_ground or vx == 0.0:
            return
        x1, _, x2, _ = deadly_rect
        if vx > 0.0 and x <= x1:
            distance = x1 - x
            speed = vx
        elif vx < 0.0 and x >= x2:
            distance = x - x2
            speed = -vx
        else:
            return
        if distance <= margin:
            risks.append((distance / max(speed, 1.0), jump_action))

    add_risk(
        x=env.fire_x,
        vx=env.fire_vx,
        on_ground=env.fire_on_ground,
        deadly_rect=env.water_rect,
        jump_action=3,
    )
    add_risk(
        x=env.water_x,
        vx=env.water_vx,
        on_ground=env.water_on_ground,
        deadly_rect=env.lava_rect,
        jump_action=6,
    )

    if risks:
        return min(risks)[1]
    return int(proposed_action)
