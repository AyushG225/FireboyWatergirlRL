"""Evaluate temple controllers on held-out test layouts.

Test layouts start at seed 100000. Training uses seeds below 90000 and
checkpoint selection uses 90000 to 99999, so this script is the only place
test seeds are touched. Every controller is measured by the same batched
rollout code in ``firewater.temple_evaluation``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
from firewater.temple_env import TempleEnv
from firewater.temple_evaluation import evaluate_temple_expert, evaluate_temple_policy

TEST_SEED_START = 100_000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="*", type=Path)
    parser.add_argument("--expert", action="store_true")
    parser.add_argument("--count", type=int, default=1_000)
    parser.add_argument("--seed-start", type=int, default=TEST_SEED_START)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be at least 1")
    if not args.expert and not args.models:
        raise SystemExit("pass --expert and/or at least one checkpoint")
    seeds = list(range(args.seed_start, args.seed_start + args.count))
    report = {}
    if args.expert:
        result = evaluate_temple_expert(seeds)
        print(result.summary())
        report["scripted expert"] = result.to_dict()
    for path in args.models:
        model = PPO.load(path, device="cpu")
        try:
            TempleEnv.mode_for_size(model.observation_space.shape[0])
        except ValueError as exc:
            raise SystemExit(f"{path}: {exc}") from exc
        result = evaluate_temple_policy(model, seeds, controller=str(path))
        print(result.summary())
        report[str(path)] = result.to_dict()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
