from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)

from firewater.evaluation import observation_mode_for_model
from firewater.generalization import (
    evaluate_generalist,
    evaluate_planner,
    format_generalization,
)
from firewater.procedural_levels import held_out_seeds


def resolve_checkpoint(path: Path) -> Path:
    if path.exists():
        return path
    zipped = path.with_suffix(".zip")
    if zipped.exists():
        return zipped
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate planners or PPO policies on unseen generated levels"
    )
    parser.add_argument("models", nargs="*", type=Path)
    parser.add_argument("--planner", action="store_true")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=100_000)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--safety-shield",
        action="store_true",
        help="Override only actions that would enter the wrong pool imminently",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be at least 1")
    if not args.planner and not args.models:
        raise SystemExit("pass --planner and/or at least one model checkpoint")

    seeds = held_out_seeds(args.count, start=args.seed_start)
    report = {}
    if args.planner:
        result = evaluate_planner(seeds)
        print(f"planner\n{format_generalization(result)}")
        report["planner"] = result.to_dict()

    for requested in args.models:
        try:
            checkpoint = resolve_checkpoint(requested)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
        model = PPO.load(checkpoint)
        mode = observation_mode_for_model(model)
        if mode != "generalized":
            raise SystemExit(
                f"{checkpoint} uses {mode!r} observations; unseen evaluation "
                "requires a 64-input generalized checkpoint"
            )
        result = evaluate_generalist(
            model,
            seeds,
            deterministic=not args.stochastic,
            policy_seed=args.policy_seed,
            safety_shield=args.safety_shield,
        )
        print(f"\n{checkpoint}\n{format_generalization(result)}")
        report[str(checkpoint)] = result.to_dict()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
