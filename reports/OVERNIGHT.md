# Overnight improvement report

Date: 2026-07-22

The strongest pre-existing deterministic checkpoint solved two of five levels.
The final geometry-aware PPO checkpoint solves every level deterministically
and succeeds on 474 of 500 reproducibly sampled episodes.

## Results

| Level | Best pre-existing deterministic | Refined deterministic | Refined stochastic |
| ---: | ---: | ---: | ---: |
| 0 | 100% | 100% | 96% |
| 1 | 100% | 100% | 98% |
| 2 | 0% | 100% | 93% |
| 3 | 0% | 100% | 93% |
| 4 | 0% | 100% | 94% |
| **Mean** | **40%** | **100%** | **94.8%** |

The deterministic result uses 20 episodes per level. Because the environment
and greedy policy are deterministic, these repeat the same trajectory and
primarily verify lifecycle consistency. The stochastic result uses 100 episodes
per level with seed 0 and resets all policy/environment random generators.

Raw data:

- [`baseline.json`](baseline.json): pre-existing checkpoint evaluation
- [`refined_deterministic.json`](refined_deterministic.json): phase-6 and final checkpoints
- [`refined_stochastic.json`](refined_stochastic.json): 500 sampled episodes per checkpoint

## What changed

- Fixed rendering, terminal lifecycle, validation, metadata, and recorder behavior.
- Corrected cooperative distance shaping and penalized partial-door/death exploits.
- Expanded observations from 19 values to 56 with goal heights, completion flags,
  level identity, platform rectangles, and full hazard boundaries.
- Preserved automatic compatibility with old 19- and 48-input checkpoints and
  replayed legacy demonstrations into the new state representation.
- Added successful scripted demonstrations for all five levels and behavior-cloning
  warm starts.
- Reworked training into a resumable seven-phase curriculum with checkpointing,
  per-level rollout diagnostics, hard-level refinement, and final consolidation.
- Made deterministic and stochastic evaluation reproducible and machine-readable.
- Added automated environment, reward, rendering, demonstration, curriculum, and
  evaluation regressions plus a CI training smoke test.

## Reproduce

The local winning artifact is `checkpoints/firewater_refined_final.zip`.

```bash
python evaluate_agent.py checkpoints/firewater_refined_final --episodes 20
python evaluate_agent.py checkpoints/firewater_refined_final \
  --episodes 100 --stochastic
python rollout_trained.py --model checkpoints/firewater_refined_final --level 4
```

To train the revised curriculum from scratch:

```bash
python train_ppo.py --run-name firewater_ppo
```
