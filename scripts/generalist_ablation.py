"""Score every generalist training run on the held-out test layouts.

Reads the ``*_validation.json`` files written by ``train_generalist.py``. For
each run it evaluates the checkpoint chosen on validation, with and without
the safety shield, and for behavior-cloned runs also the policy before PPO.
Test seeds are touched only here, after every selection is already fixed.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
from firewater.procedural_levels import held_out_seeds


def _evaluate(job: dict) -> dict:
    import torch
    from stable_baselines3 import PPO

    from firewater.generalization import evaluate_generalist, evaluate_planner

    torch.set_num_threads(1)
    seeds = held_out_seeds(job["count"], start=job["seed_start"])
    if job["variant"] == "planner":
        result = evaluate_planner(seeds)
    else:
        model = PPO.load(job["checkpoint"], device="cpu")
        result = evaluate_generalist(
            model, seeds, safety_shield=job["variant"].endswith("shield")
        )
    return job | result.to_dict()


def _jobs(runs_dir: Path, count: int, seed_start: int) -> list[dict]:
    jobs = []
    for report_path in sorted(runs_dir.glob("*_validation.json")):
        report = json.loads(report_path.read_text())
        config = "bc_ppo" if report["behavior_cloning"] else "ppo_only"
        selected = runs_dir / f"{report['selected_checkpoint']}.zip"
        start = runs_dir / f"{report['history'][0]['checkpoint']}.zip"
        base = {
            "config": config,
            "training_seed": report["seed"],
            "count": count,
            "seed_start": seed_start,
        }
        variants = [("policy", selected), ("policy_shield", selected)]
        if report["behavior_cloning"]:
            variants.insert(0, ("bc_only", start))
        for variant, checkpoint in variants:
            jobs.append(base | {"variant": variant, "checkpoint": str(checkpoint)})
    return jobs


def _aggregate(results: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for result in results:
        groups.setdefault((result["config"], result["variant"]), []).append(result)
    summary = []
    for (config, variant), runs in sorted(groups.items()):
        rates = np.array([run["success_rate"] for run in runs])
        per_difficulty = {}
        for difficulty in runs[0]["per_difficulty"]:
            counts = [run["per_difficulty"][difficulty] for run in runs]
            per_difficulty[difficulty] = {
                key: float(np.mean([count[key] for count in counts]))
                for key in ("layouts", "successes", "hazards", "timeouts")
            }
        summary.append(
            {
                "config": config,
                "variant": variant,
                "runs": len(runs),
                "success_rate_mean": float(rates.mean()),
                "success_rate_std": float(rates.std(ddof=1)) if len(runs) > 1 else 0.0,
                "success_rate_min": float(rates.min()),
                "success_rate_max": float(rates.max()),
                "hazard_rate_mean": float(
                    np.mean([run["hazards"] / run["layouts"] for run in runs])
                ),
                "timeout_rate_mean": float(
                    np.mean([run["timeouts"] / run["layouts"] for run in runs])
                ),
                "mean_length_mean": float(np.mean([run["mean_length"] for run in runs])),
                "per_difficulty_mean_counts": per_difficulty,
            }
        )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("checkpoints/generalist_v2")
    )
    parser.add_argument("--count", type=int, default=1_000)
    parser.add_argument("--seed-start", type=int, default=100_000)
    parser.add_argument("--planner", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    jobs = _jobs(args.runs_dir, args.count, args.seed_start)
    if args.planner:
        jobs.append(
            {
                "config": "planner",
                "variant": "planner",
                "training_seed": None,
                "count": args.count,
                "seed_start": args.seed_start,
            }
        )
    if not jobs:
        raise SystemExit(f"no *_validation.json files in {args.runs_dir}")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_evaluate, jobs))
    for result in results:
        print(
            f"{result['config']:9s} {result['variant']:14s} "
            f"seed={result['training_seed']} success={result['success_rate']:.1%} "
            f"hazard={result['hazards'] / result['layouts']:.1%} "
            f"timeout={result['timeouts'] / result['layouts']:.1%}"
        )
    report = {
        "test_seeds": [args.seed_start, args.seed_start + args.count - 1],
        "selection": "each run's checkpoint was chosen on validation seeds 90000+",
        "summary": _aggregate(results),
        "runs": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
