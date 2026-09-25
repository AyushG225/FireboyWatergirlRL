# Unseen-level generalization report

Environment: `FireWaterEnv` in procedural mode. One `Discrete(7)` action per
frame moves one character, and the policy sees a 64-value geometry observation
with no level ID. A seed deterministically generates a complete layout: spawn
points, door positions and heights, pool positions and widths, gaps, and
platforms.

## Results on 1,000 test layouts

All rows use test seeds 100000 to 100999. Each learned configuration was
trained three times (training seeds 0, 1, 2) and every run's checkpoint was
chosen on validation layouts before any test layout was played.

| Controller | Type | Success, mean of 3 runs | Range | Hazard deaths | Timeouts |
| --- | --- | ---: | ---: | ---: | ---: |
| Behavior cloning + PPO | neural policy | **97.3%** (sd 1.7) | 95.4% to 98.4% | 0.3% | 2.3% |
| Behavior cloning + PPO + safety shield | neural policy + shield | 95.3% (sd 1.4) | 94.0% to 96.8% | 1.4% | 3.3% |
| Behavior cloning only | neural policy | 0.1% | 0.0% to 0.3% | 98.8% | 1.1% |
| PPO only, no cloning | neural policy | 0.0% | 0.0% to 0.0% | 56.1% | 43.9% |
| Weighted A* planner | planner | 100.0% | single run | 0.0% | 0.0% |

Source: [`generalist_ablation_1000.json`](generalist_ablation_1000.json).

Neither ingredient works alone. Cloning 300 planner demonstrations produces a
policy that walks into a pool within about 49 frames on nearly every layout.
PPO from random weights never solves a validation layout: none of the 42
PPO-only checkpoints (14 per run, saved every 250,000 steps) completes one of
the 300 validation layouts. PPO starting from the cloned policy reaches 97.3%.

### Per-difficulty breakdown

The generator draws one of three difficulty families per seed. Counts are
means over the three runs.

| Difficulty | Layouts | BC + PPO success | BC + PPO + shield success | Planner success |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 333 | 332.0 (99.7%) | 333.0 (100.0%) | 333 (100%) |
| 1 | 334 | 329.7 (98.7%) | 323.0 (96.7%) | 334 (100%) |
| 2 | 333 | 311.7 (93.6%) | 297.0 (89.2%) | 333 (100%) |

Difficulty 2 accounts for most policy failures, and most of those are
timeouts (20.7 of 21.3 failed layouts on average).

### Why the safety shield lowers success

The shield replaces the policy's action with a jump when a grounded character
is moving toward the pool that kills it and is at most 70 px from its edge. A
paired comparison on 1,000 validation layouts (seeds 90000 to 90999) played
each layout twice with the same deterministic policy, once plain and once
shielded:

| Run | Failures the shield turned into successes | Successes the shield turned into failures | Shield overrides in those broken episodes |
| --- | ---: | ---: | ---: |
| training seed 0 | 5 | 23 | 1 to 2 |
| training seed 1 | 8 | 51 | 1 to 4 |
| training seed 2 | 4 | 20 | 1 to 2 |

Source: [`shield_paired_validation.json`](shield_paired_validation.json).

The trained policies time their own jumps closer to the pool edge. One or two
forced early jumps are enough to land short or leave a character off its
route. The shield was designed around an earlier checkpoint that walked into
pools on 38 of 1,000 test layouts; for that checkpoint it raised success to
100% (see the archive below).

## Protocol

| Split | Seeds | Used for |
| --- | --- | --- |
| Training | 0 to 89,999 | planner demonstrations (seeds 0 to 299) and PPO environments |
| Validation | 90,000 to 99,999 | choosing each run's checkpoint (seeds 90000 to 90299); shield diagnosis (seeds 90000 to 90999) |
| Test | 100,000 and above | the table above, evaluated once per configuration |

- Every checkpoint saved during PPO (every 250,000 steps, plus the final one)
  was scored on 300 validation layouts. The rule, fixed before training, picks
  the highest success, then fewer hazards, then the earlier checkpoint.
- Selected checkpoints: 2,750,000 steps (seed 0, 98.0% on validation),
  3,000,000 steps (seed 1, 98.7%), and 3,000,000 steps (seed 2, 96.3%).
  Histories: [`generalist_v2_validation/`](generalist_v2_validation).
- Training: 8 environments, 3,000,000 PPO steps, 256 by 256 MLP, learning rate
  1e-4, entropy 0.002, clip 0.1. Cloning used 20 epochs over the demonstrations
  from training seeds 0 to 299.
- Evaluation is deterministic, runs each layout once, and resets model and
  environment random generators.
- Every test layout is independently verified solvable by the planner.

## Revision: earlier 96.1% result

An earlier version of this report listed 96.1% for a single checkpoint. That
checkpoint was chosen from 13 candidates by scoring them on seeds 100000 to
100299, which are the first 300 of the 1,000 test layouts, and its training
run sampled layouts from the full seed space. On the 700 test layouts outside
the selection set it solved 675 (96.4%). The runs above replace it. Its raw
files are in [`archive/generalist_v1/`](archive/generalist_v1):
[`generalist_policy_1000.json`](archive/generalist_v1/generalist_policy_1000.json)
(961 of 1,000),
[`generalist_v1_checkpoints_300.json`](archive/generalist_v1/generalist_v1_checkpoints_300.json)
(286 of 300 on the selection seeds), and
[`generalist_hybrid_1000.json`](archive/generalist_v1/generalist_hybrid_1000.json)
(with the shield).

## Scope

This measures generalization to newly generated layouts within the mechanics
of `FireWaterEnv`. It does not show that the controller can operate levels
from the commercial browser game, whose sprites, switches, elevators, camera,
and physics this simulator does not model.

## Reproduce

```bash
for seed in 0 1 2; do
  python scripts/train_generalist.py --seed $seed --run-name gen_bc_s$seed \
    --output-dir checkpoints/generalist_v2
  python scripts/train_generalist.py --seed $seed --no-bc --run-name gen_nobc_s$seed \
    --output-dir checkpoints/generalist_v2
done
python scripts/generalist_ablation.py --planner \
  --output reports/generalist_ablation_1000.json
python scripts/shield_diagnosis.py --output reports/shield_paired_validation.json
```
