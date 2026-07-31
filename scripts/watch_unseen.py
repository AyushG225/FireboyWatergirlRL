from __future__ import annotations

import argparse
import secrets
from pathlib import Path

from stable_baselines3 import PPO

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)

from firewater.firewater_env import FireWaterEnv
from firewater.generalized_planner import plan_level
from firewater.safety_shield import shield_action


UNSEEN_SEED_START = 100_000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a brand-new level and watch both players solve it"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--no-safety-shield",
        action="store_true",
        help="Disable wrong-pool prevention when using a neural model",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    layout_seed = (
        args.seed
        if args.seed is not None
        else UNSEEN_SEED_START
        + secrets.randbelow(2**31 - UNSEEN_SEED_START)
    )
    model = PPO.load(args.model) if args.model is not None else None

    if model is None:
        planning_env = FireWaterEnv(
            procedural=True,
            level_seed=layout_seed,
            observation_mode="generalized",
        )
        planning_env.reset(seed=0)
        try:
            plan = plan_level(planning_env)
        finally:
            planning_env.close()
        actions = iter(plan.actions)
        print(
            f"new layout seed={layout_seed}; geometry planner found a "
            f"{len(plan.actions)}-step solution"
        )
    else:
        actions = None
        print(f"new layout seed={layout_seed}; policy={args.model}")

    env = FireWaterEnv(
        render_mode="human",
        procedural=True,
        level_seed=layout_seed,
        observation_mode="generalized",
    )
    observation, _ = env.reset(seed=0)
    try:
        for step_index in range(env.max_steps):
            if model is None:
                try:
                    action = next(actions)
                except StopIteration:
                    break
            else:
                action, _ = model.predict(
                    observation,
                    deterministic=not args.stochastic,
                )
                if not args.no_safety_shield:
                    action = shield_action(env, action)
            observation, _, terminated, truncated, info = env.step(action)
            if env.window_closed:
                return
            if terminated or truncated:
                print(
                    f"seed={layout_seed} reason={info['reason']} "
                    f"steps={step_index + 1}"
                )
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
