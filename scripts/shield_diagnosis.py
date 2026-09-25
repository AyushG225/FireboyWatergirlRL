"""Paired comparison of each policy with and without the safety shield.

Every layout is played twice by the same deterministic policy, once plain and
once shielded, and the pair of outcomes is tallied. Uses validation seeds, so
the analysis never looks at the test layouts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)


def _play(model, seed: int, shield: bool) -> tuple[str, int]:
    from firewater.firewater_env import FireWaterEnv
    from firewater.safety_shield import shield_action

    env = FireWaterEnv(procedural=True, level_seed=seed, observation_mode="generalized")
    observation, _ = env.reset(seed=0)
    overrides = 0
    info = {"reason": None}
    for _ in range(env.max_steps):
        action, _ = model.predict(observation, deterministic=True)
        action = int(action)
        if shield:
            shielded = shield_action(env, action)
            overrides += int(shielded != action)
            action = shielded
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()
    return str(info["reason"]), overrides


def _diagnose(job: dict) -> dict:
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    model = PPO.load(job["checkpoint"], device="cpu")
    pairs: Counter = Counter()
    broken_overrides = []
    for seed in range(job["seed_start"], job["seed_start"] + job["count"]):
        plain, _ = _play(model, seed, shield=False)
        shielded, overrides = _play(model, seed, shield=True)
        pairs[f"{plain} -> {shielded}"] += 1
        if plain == "success" and shielded != "success":
            broken_overrides.append(overrides)
    return job | {
        "pairs": dict(pairs.most_common()),
        "rescued": sum(
            count
            for key, count in pairs.items()
            if not key.startswith("success") and key.endswith("success")
        ),
        "broken": len(broken_overrides),
        "overrides_in_broken_episodes": {
            "min": min(broken_overrides, default=0),
            "max": max(broken_overrides, default=0),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("checkpoints/generalist_v2")
    )
    parser.add_argument("--count", type=int, default=1_000)
    parser.add_argument("--seed-start", type=int, default=90_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed_start + args.count > 100_000:
        raise SystemExit("shield diagnosis must stay on validation seeds (< 100000)")
    jobs = []
    for report_path in sorted(args.runs_dir.glob("*_validation.json")):
        report = json.loads(report_path.read_text())
        if report["behavior_cloning"]:
            jobs.append(
                {
                    "run": report_path.name.removesuffix("_validation.json"),
                    "checkpoint": str(
                        args.runs_dir / f"{report['selected_checkpoint']}.zip"
                    ),
                    "seed_start": args.seed_start,
                    "count": args.count,
                }
            )
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        results = list(pool.map(_diagnose, jobs))
    for result in results:
        print(
            f"{result['run']}: rescued={result['rescued']} broken={result['broken']} "
            f"overrides_in_broken={result['overrides_in_broken_episodes']}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "validation_seeds": [args.seed_start, args.seed_start + args.count - 1],
                "runs": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
