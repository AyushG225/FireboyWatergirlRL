"""Cooperative two-character reinforcement learning environments and training.

The package is organized in three layers:

- Environments: ``firewater_env`` and ``temple_env``, with their level
  generators ``procedural_levels`` and ``temple_levels``.
- Experts: ``scripted_demos``, ``generalized_planner``, ``temple_expert``, and
  the level-agnostic ``safety_shield``.
- Training and evaluation: ``train_ppo``, ``evaluation``, and
  ``generalization``.

Runnable commands live in the top-level ``scripts/`` directory.
"""
