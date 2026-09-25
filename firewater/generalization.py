from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
from stable_baselines3.common.utils import set_random_seed

from firewater.firewater_env import FireWaterEnv
from firewater.generalized_planner import plan_level
from firewater.procedural_levels import generate_level
from firewater.safety_shield import shield_action


@dataclass(frozen=True)
class GeneralizationEvaluation:
    layouts: int
    successes: int
    hazards: int
    timeouts: int
    mean_return: float
    mean_length: float
    per_difficulty_success: dict[int, float]
    # difficulty -> {"layouts", "successes", "hazards", "timeouts"}
    per_difficulty: dict[int, dict[str, int]]

    @property
    def success_rate(self) -> float:
        return self.successes / self.layouts

    def to_dict(self):
        result = asdict(self)
        result["success_rate"] = self.success_rate
        return result


def _difficulty_breakdown(
    outcomes: dict[int, list[str]],
) -> tuple[dict[int, float], dict[int, dict[str, int]]]:
    success_rates = {
        difficulty: (
            float(np.mean([reason == "success" for reason in reasons]))
            if reasons
            else 0.0
        )
        for difficulty, reasons in outcomes.items()
    }
    counts = {
        difficulty: {
            "layouts": len(reasons),
            "successes": reasons.count("success"),
            "hazards": reasons.count("hazard"),
            "timeouts": reasons.count("timeout"),
        }
        for difficulty, reasons in outcomes.items()
    }
    return success_rates, counts


def evaluate_generalist(
    model,
    seeds: Iterable[int],
    *,
    deterministic: bool = True,
    policy_seed: int = 0,
    safety_shield: bool = False,
) -> GeneralizationEvaluation:
    """Evaluate one policy once on each held-out procedural layout."""
    seeds = [int(seed) for seed in seeds]
    if not seeds:
        raise ValueError("at least one layout seed is required")
    set_random_seed(policy_seed)

    terminal_counts = {"success": 0, "hazard": 0, "timeout": 0}
    returns = []
    lengths = []
    difficulty_outcomes: dict[int, list[str]] = {0: [], 1: [], 2: []}

    for layout_seed in seeds:
        env = FireWaterEnv(
            procedural=True,
            level_seed=layout_seed,
            observation_mode="generalized",
        )
        observation, _ = env.reset(seed=policy_seed)
        episode_return = 0.0
        info = {"reason": None}
        try:
            for _ in range(env.max_steps):
                action, _ = model.predict(
                    observation,
                    deterministic=deterministic,
                )
                if safety_shield:
                    action = shield_action(env, action)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                if terminated or truncated:
                    break
            episode_length = env.steps
        finally:
            env.close()

        reason = info.get("reason")
        if reason in terminal_counts:
            terminal_counts[reason] += 1
        returns.append(episode_return)
        lengths.append(episode_length)
        difficulty = generate_level(layout_seed).difficulty
        difficulty_outcomes[difficulty].append(str(reason))

    success_rates, counts = _difficulty_breakdown(difficulty_outcomes)
    return GeneralizationEvaluation(
        layouts=len(seeds),
        successes=terminal_counts["success"],
        hazards=terminal_counts["hazard"],
        timeouts=terminal_counts["timeout"],
        mean_return=float(np.mean(returns)),
        mean_length=float(np.mean(lengths)),
        per_difficulty_success=success_rates,
        per_difficulty=counts,
    )


def evaluate_planner(seeds: Iterable[int]) -> GeneralizationEvaluation:
    """Evaluate the geometry planner on held-out layouts."""
    seeds = [int(seed) for seed in seeds]
    if not seeds:
        raise ValueError("at least one layout seed is required")

    returns = []
    lengths = []
    difficulty_outcomes: dict[int, list[str]] = {0: [], 1: [], 2: []}
    for layout_seed in seeds:
        env = FireWaterEnv(
            procedural=True,
            level_seed=layout_seed,
            observation_mode="generalized",
        )
        env.reset(seed=0)
        try:
            result = plan_level(env)
            # plan_level leaves the verified environment at its terminal state.
            success = env._both_at_goals()
            returns.append(result.total_reward)
            lengths.append(len(result.actions))
        finally:
            env.close()
        difficulty = generate_level(layout_seed).difficulty
        difficulty_outcomes[difficulty].append("success" if success else "failure")

    success_rates, counts = _difficulty_breakdown(difficulty_outcomes)
    return GeneralizationEvaluation(
        layouts=len(seeds),
        successes=sum(count["successes"] for count in counts.values()),
        hazards=0,
        timeouts=0,
        mean_return=float(np.mean(returns)),
        mean_length=float(np.mean(lengths)),
        per_difficulty_success=success_rates,
        per_difficulty=counts,
    )


def format_generalization(result: GeneralizationEvaluation) -> str:
    difficulty_text = " ".join(
        f"difficulty-{difficulty}={rate:.1%}"
        for difficulty, rate in sorted(result.per_difficulty_success.items())
    )
    return (
        f"layouts={result.layouts} success={result.success_rate:.1%} "
        f"hazard={result.hazards / result.layouts:.1%} "
        f"timeout={result.timeouts / result.layouts:.1%} "
        f"mean_return={result.mean_return:.3f} "
        f"mean_length={result.mean_length:.1f} [{difficulty_text}]"
    )
