from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
from firewater.evaluation import evaluate_model, format_evaluation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one or more PPO checkpoints on every level"
    )
    parser.add_argument("models", nargs="+", help="PPO checkpoint paths")
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        choices=range(5),
        default=list(range(5)),
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample policy actions instead of using deterministic actions",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser.parse_args()


def resolve_checkpoint(path: str) -> Path:
    checkpoint = Path(path)
    if checkpoint.exists():
        return checkpoint
    zipped = checkpoint.with_suffix(".zip")
    if zipped.exists():
        return zipped
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def main():
    args = parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1")

    set_random_seed(args.seed)
    report = {}

    for requested_path in args.models:
        try:
            checkpoint = resolve_checkpoint(requested_path)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc

        model = PPO.load(checkpoint)
        results = evaluate_model(
            model,
            levels=args.levels,
            episodes=args.episodes,
            deterministic=not args.stochastic,
            seed=args.seed,
        )

        print(f"\n{checkpoint}")
        print(format_evaluation(results))
        report[str(checkpoint)] = {
            str(level_id): result.to_dict() for level_id, result in results.items()
        }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote evaluation report to {args.output}")


if __name__ == "__main__":
    main()
