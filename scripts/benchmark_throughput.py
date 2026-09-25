"""Measure raw environment throughput with random actions.

Reports steps per second for one environment and for Stable-Baselines3
vectorized environments (in-process DummyVecEnv and multi-process
SubprocVecEnv). No policy inference or gradient updates are included, so the
numbers are an upper bound on simulation speed during training.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
from firewater.firewater_env import FireWaterEnv
from firewater.temple_env import TempleEnv


def make_env(name: str, rank: int):
    def _init():
        if name == "firewater":
            env = FireWaterEnv(
                procedural=True,
                observation_mode="generalized",
                procedural_seed_range=(0, 100_000),
            )
        else:
            env = TempleEnv(seed_range=(0, 100_000))
        env.reset(seed=rank)
        env.action_space.seed(rank)
        return env

    return _init


def time_single(name: str, steps: int) -> float:
    env = make_env(name, 0)()
    rng = np.random.default_rng(0)
    n_actions = env.action_space.n
    for _ in range(1_000):
        _, _, terminated, truncated, _ = env.step(int(rng.integers(n_actions)))
        if terminated or truncated:
            env.reset()
    start = time.perf_counter()
    for _ in range(steps):
        _, _, terminated, truncated, _ = env.step(int(rng.integers(n_actions)))
        if terminated or truncated:
            env.reset()
    elapsed = time.perf_counter() - start
    env.close()
    return steps / elapsed


def time_vectorized(name: str, kind: str, n_envs: int, steps: int) -> float:
    factories = [make_env(name, rank) for rank in range(n_envs)]
    vec_env = DummyVecEnv(factories) if kind == "dummy" else SubprocVecEnv(factories)
    rng = np.random.default_rng(0)
    n_actions = vec_env.action_space.n
    vec_env.reset()
    for _ in range(200):
        vec_env.step(rng.integers(n_actions, size=n_envs))
    batches = max(steps // n_envs, 1)
    start = time.perf_counter()
    for _ in range(batches):
        vec_env.step(rng.integers(n_actions, size=n_envs))
    elapsed = time.perf_counter() - start
    vec_env.close()
    return batches * n_envs / elapsed


def cpu_model() -> str:
    """Best-effort CPU name for the report."""
    try:
        if platform.system() == "Darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return platform.processor() or "unknown"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, nargs="+", default=[8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    results = []

    def record(name: str, mode: str, n_envs: int, measure, *measure_args) -> None:
        samples = [measure(*measure_args) for _ in range(args.repeats)]
        median = float(np.median(samples))
        results.append(
            {
                "env": name,
                "mode": mode,
                "n_envs": n_envs,
                "steps_per_sec_median": median,
                "steps_per_sec_samples": samples,
            }
        )
        label = mode if n_envs == 1 else f"{mode} x{n_envs}"
        print(f"{name:9s} {label:22s} {median:12,.0f} steps/s (median of {args.repeats})")

    for name in ("firewater", "temple"):
        record(name, "single", 1, time_single, name, args.steps)
        for n_envs in args.n_envs:
            for kind in ("dummy", "subproc"):
                record(
                    name,
                    f"{kind}_vec_env",
                    n_envs,
                    time_vectorized,
                    name,
                    kind,
                    n_envs,
                    args.steps,
                )

    if args.output is not None:
        report = {
            "machine": {
                "platform": platform.platform(),
                "cpu": cpu_model(),
                "cpu_count": os.cpu_count(),
                "python": platform.python_version(),
            },
            "steps_per_measurement": args.steps,
            "repeats": args.repeats,
            "actions": "uniform random",
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
