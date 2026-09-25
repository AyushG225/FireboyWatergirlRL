from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
from stable_baselines3.common.utils import set_random_seed

from firewater.firewater_env import FireWaterEnv


@dataclass(frozen=True)
class LevelEvaluation:
    level_id: int
    episodes: int
    successes: int
    hazards: int
    timeouts: int
    mean_return: float
    std_return: float
    mean_length: float

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes

    @property
    def hazard_rate(self) -> float:
        return self.hazards / self.episodes

    @property
    def timeout_rate(self) -> float:
        return self.timeouts / self.episodes

    def to_dict(self) -> dict[str, int | float]:
        result = asdict(self)
        result.update(
            success_rate=self.success_rate,
            hazard_rate=self.hazard_rate,
            timeout_rate=self.timeout_rate,
        )
        return result


def observation_mode_for_model(model) -> str:
    """Select the compatible environment observation mode for a model."""
    observation_space = getattr(model, "observation_space", None)
    shape = getattr(observation_space, "shape", None)
    if shape == (FireWaterEnv.ENHANCED_OBSERVATION_SIZE,):
        return "enhanced"
    if shape == (FireWaterEnv.GENERALIZED_OBSERVATION_SIZE,):
        return "generalized"
    if shape == (FireWaterEnv.ENHANCED_V1_OBSERVATION_SIZE,):
        return "enhanced_v1"
    if shape in (None, (FireWaterEnv.LEGACY_OBSERVATION_SIZE,)):
        return "legacy"
    raise ValueError(f"Unsupported model observation shape: {shape}")


def evaluate_model(
    model,
    levels: Iterable[int] = range(5),
    episodes: int = 10,
    deterministic: bool = True,
    seed: int = 0,
) -> dict[int, LevelEvaluation]:
    """Evaluate an SB3-compatible model and retain terminal-reason metrics."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    # Stochastic policies draw actions from PyTorch's global generator. Reset
    # all supported RNGs here so repeated reports and multi-checkpoint
    # comparisons use the same action-sampling stream.
    set_random_seed(seed)

    results = {}
    for level_id in levels:
        env = FireWaterEnv(
            level_id=int(level_id),
            observation_mode=observation_mode_for_model(model),
        )
        returns = []
        lengths = []
        terminal_counts = {"success": 0, "hazard": 0, "timeout": 0}

        try:
            for episode_idx in range(episodes):
                observation, _ = env.reset(seed=seed + episode_idx)
                episode_return = 0.0

                for step_idx in range(env.max_steps):
                    action, _ = model.predict(
                        observation,
                        deterministic=deterministic,
                    )
                    observation, reward, terminated, truncated, info = env.step(action)
                    episode_return += reward

                    if terminated or truncated:
                        reason = info.get("reason")
                        if reason in terminal_counts:
                            terminal_counts[reason] += 1
                        returns.append(episode_return)
                        lengths.append(step_idx + 1)
                        break
                else:  # pragma: no cover - max_steps should always truncate
                    returns.append(episode_return)
                    lengths.append(env.max_steps)
        finally:
            env.close()

        results[int(level_id)] = LevelEvaluation(
            level_id=int(level_id),
            episodes=episodes,
            successes=terminal_counts["success"],
            hazards=terminal_counts["hazard"],
            timeouts=terminal_counts["timeout"],
            mean_return=float(np.mean(returns)),
            std_return=float(np.std(returns)),
            mean_length=float(np.mean(lengths)),
        )

    return results


def format_evaluation(results: dict[int, LevelEvaluation]) -> str:
    header = "level  success  hazard  timeout  mean return  mean length"
    rows = [header]
    for level_id in sorted(results):
        result = results[level_id]
        rows.append(
            f"{level_id:>5}  "
            f"{result.success_rate:>7.1%}  "
            f"{result.hazard_rate:>6.1%}  "
            f"{result.timeout_rate:>7.1%}  "
            f"{result.mean_return:>11.3f}  "
            f"{result.mean_length:>11.1f}"
        )
    return "\n".join(rows)
