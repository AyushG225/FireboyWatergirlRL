from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from firewater.firewater_env import FireWaterEnv


@dataclass(frozen=True)
class Demonstration:
    level_id: int
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    reason: str


def _append_step(env, action, observations, actions, rewards):
    observations.append(env._get_obs().copy())
    actions.append(int(action))
    _, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)
    return terminated or truncated, info


def _directional_action(env, who: str, jump_margin: float) -> int:
    if who == "fire":
        x = env.fire_x
        on_ground = env.fire_on_ground
        goal_x, goal_y = env.fire_goal_x, env.fire_goal_y
        deadly_pool = env.water_rect
        jump_action = 3
        left_action, right_action = 1, 2
    else:
        x = env.water_x
        on_ground = env.water_on_ground
        goal_x, goal_y = env.water_goal_x, env.water_goal_y
        deadly_pool = env.lava_rect
        jump_action = 6
        left_action, right_action = 4, 5

    direction = 1 if goal_x > x else -1
    move_action = right_action if direction > 0 else left_action
    pool_x1, _, pool_x2, _ = deadly_pool
    approaching_pool = (
        direction > 0 and x < pool_x1 and pool_x1 - x < jump_margin
    ) or (
        direction < 0 and x > pool_x2 and x - pool_x2 < jump_margin
    )
    approaching_raised_door = (
        goal_y < env.ground_y and abs(goal_x - x) < 150.0
    )

    if on_ground and (approaching_pool or approaching_raised_door):
        return jump_action
    return move_action


def _run_simple_level(env, observations, actions, rewards):
    # Level 2 has an overhead center platform, so its jump must start earlier.
    jump_margin = 100.0 if env.level_id in (0, 1, 2) else 50.0
    info = {"reason": None}

    while len(actions) < env.max_steps:
        who = "fire" if not env.fire_at_goal else "water"
        action = _directional_action(env, who, jump_margin)
        done, info = _append_step(env, action, observations, actions, rewards)
        if done:
            return info

    return info


def _run_actions(env, sequence, observations, actions, rewards):
    info = {"reason": None}
    for action in sequence:
        done, info = _append_step(env, action, observations, actions, rewards)
        if done:
            break
    return info


def _run_level_four(env, observations, actions, rewards):
    # These segments traverse the left-to-right platform chain for Fire, then
    # its mirrored right-to-left chain for Water. No state is teleported: every
    # transition goes through FireWaterEnv.step and is regression-tested.
    fire_segments = (
        [2, 3] + [2] * 24 + [0],
        [3] + [2] * 6 + [0] * 20,
        [3] + [2] * 3 + [0] * 24,
        [3] + [2] * 24 + [0] * 3,
    )
    water_segments = (
        [4, 6] + [4] * 24 + [0],
        [6] + [4] * 5 + [0] * 18,
        [4] * 7 + [0] * 7,
        [4] * 2 + [6] + [4] * 22 + [0] * 3,
    )

    info = {"reason": None}
    for segment in fire_segments:
        info = _run_actions(env, segment, observations, actions, rewards)
        if info["reason"] is not None:
            return info

    while not env.fire_at_goal and len(actions) < env.max_steps:
        action = 2 if env.fire_x < env.fire_goal_x else 1
        done, info = _append_step(env, action, observations, actions, rewards)
        if done:
            return info

    for segment in water_segments:
        info = _run_actions(env, segment, observations, actions, rewards)
        if info["reason"] is not None:
            return info

    while not env.water_at_goal and len(actions) < env.max_steps:
        action = 4 if env.water_x > env.water_goal_x else 5
        done, info = _append_step(env, action, observations, actions, rewards)
        if done:
            return info

    return info


def generate_scripted_demo(
    level_id: int,
    observation_mode: str = "enhanced",
) -> Demonstration:
    """Generate a deterministic successful demonstration for one level."""
    env = FireWaterEnv(level_id=level_id, observation_mode=observation_mode)
    env.reset(seed=0)
    observations = []
    actions = []
    rewards = []

    try:
        if level_id == 4:
            info = _run_level_four(env, observations, actions, rewards)
        else:
            info = _run_simple_level(env, observations, actions, rewards)

        if info.get("reason") != "success":
            raise RuntimeError(
                f"scripted controller did not solve level {level_id}: {info}"
            )

        return Demonstration(
            level_id=level_id,
            observations=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            reason=info["reason"],
        )
    finally:
        env.close()


def save_scripted_demo(demo: Demonstration, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        obs=demo.observations,
        acts=demo.actions,
        rewards=demo.rewards,
        level_id=np.asarray(demo.level_id, dtype=np.int64),
        reason=np.asarray(demo.reason),
        format_version=np.asarray(1, dtype=np.int64),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate successful scripted behavior-cloning demonstrations"
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        choices=range(5),
        default=list(range(5)),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Generate 19-value observations for legacy checkpoints",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for level_id in args.levels:
        demo = generate_scripted_demo(
            level_id,
            observation_mode="legacy" if args.legacy else "enhanced",
        )
        path = args.output_dir / f"demo_level{level_id}_scripted.npz"
        save_scripted_demo(demo, path)
        print(
            f"level={level_id} steps={len(demo.actions)} "
            f"return={demo.rewards.sum():.3f} output={path}"
        )


if __name__ == "__main__":
    main()
