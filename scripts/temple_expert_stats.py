"""Check the stateless expert and count rare actions in the demonstrations.

1. On validation layouts, run the original expert and ask the stateless expert
   for its action on every frame. Report how many frames disagree.
2. On the demonstration layouts, count how often each character is told to
   jump. These frames are what the weighted cloning loss upweights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
from firewater.temple_env import LOCAL_ACTION_COUNT, TempleEnv
from firewater.temple_expert import TempleExpert

JUMP_COMMANDS = (3, 4, 5)


def compare_experts(seeds: range) -> dict:
    frames = disagreements = successes = 0
    for seed in seeds:
        env = TempleEnv(level_seed=seed)
        env.reset(seed=0)
        original, stateless = TempleExpert(), TempleExpert(stateless=True)
        info = {"reason": None}
        for _ in range(env.max_steps):
            action = original.action(env)
            disagreements += int(stateless.action(env) != action)
            frames += 1
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        successes += int(info["reason"] == "success")
        env.close()
    return {
        "seeds": [seeds.start, seeds.stop - 1],
        "frames": frames,
        "disagreeing_frames": disagreements,
        "original_expert_successes": successes,
    }


def count_jumps(seeds: range) -> dict:
    frames = 0
    jumps = {"fire": 0, "water": 0}
    for seed in seeds:
        env = TempleEnv(level_seed=seed)
        env.reset(seed=0)
        expert = TempleExpert()
        for _ in range(env.max_steps):
            action = expert.action(env)
            fire, water = divmod(action, LOCAL_ACTION_COUNT)
            jumps["fire"] += int(fire in JUMP_COMMANDS)
            jumps["water"] += int(water in JUMP_COMMANDS)
            frames += 1
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        env.close()
    return {
        "seeds": [seeds.start, seeds.stop - 1],
        "frames": frames,
        "jump_frames": jumps,
        "jump_share": {who: count / frames for who, count in jumps.items()},
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--validation-count", type=int, default=200)
    parser.add_argument("--demo-count", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    report = {
        "stateless_vs_original": compare_experts(
            range(90_000, 90_000 + args.validation_count)
        ),
        "demonstration_jumps": count_jumps(range(args.demo_count)),
    }
    print(json.dumps(report, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
