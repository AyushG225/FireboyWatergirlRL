# Unseen-level generalization report

Date: 2026-07-22

## Outcome

One controller now handles layouts that were not part of the original five
levels and whose exact seeds never appeared during training.

| Controller | Held-out layouts | Success | Hazards | Timeouts |
| --- | ---: | ---: | ---: | ---: |
| Geometry-only PPO | 1,000 | 96.1% | 3.8% | 0.1% |
| PPO + wrong-pool safety shield | 1,000 | 100% | 0% | 0% |
| Weighted-A* geometry planner | 1,000 | 100% | 0% | 0% |

Each difficulty family independently reaches 100% with the shield and planner.
The shield changes an action only when a grounded player is moving toward its
deadly pool and is at most 70 pixels from the near edge.

## Protocol

- Behavior-cloning demonstrations used procedural seeds `0–299`.
- PPO generated 18,615 distinct training episodes.
- Training logs for the selected checkpoint contain none of the 1,000
  benchmark seeds.
- The benchmark uses seeds `100000–100999`.
- The selected checkpoint predates the bounded sampler and used the full
  procedural seed space, but its recorded layouts have zero exact overlap with
  the benchmark. The procedural trainer now enforces the stronger disjoint
  `[0, 100000)` training range for future runs.
- The model receives a 64-value geometry observation with no level ID.
- Every benchmark layout is independently verified solvable by the planner.
- Evaluation is deterministic and resets model/environment random generators.

Raw reports:

- [`generalist_policy_1000.json`](generalist_policy_1000.json)
- [`generalist_hybrid_1000.json`](generalist_hybrid_1000.json)
- [`generalist_planner_1000.json`](generalist_planner_1000.json)
- [`generalist_v1_checkpoints_300.json`](generalist_v1_checkpoints_300.json)

The selected neural artifact is
`checkpoints/firewater_generalist_best.zip`, copied from the two-million-step
checkpoint after it scored 95.3% on a separate 300-layout selection suite.

## What “new level” means

The generator changes the complete layout: player spawns, door positions and
heights, pool positions, pool widths, gaps, platform widths, and decorative
platform geometry. A seed deterministically identifies a layout, which makes
train/test separation and failure reproduction exact.

This is genuine layout generalization within the mechanics implemented by
`FireWaterEnv`. It is not yet a claim that the controller can parse and operate
an arbitrary level from the commercial browser game, whose sprites, switches,
elevators, buttons, camera, and physics are not represented in this simulator.

## Reproduce

```bash
python evaluate_generalist.py checkpoints/firewater_generalist_best \
  --count 1000
python evaluate_generalist.py checkpoints/firewater_generalist_best \
  --count 1000 --safety-shield
python evaluate_generalist.py --planner --count 1000
python watch_unseen.py
```
