# Complex temple environment

Date: 2026-07-22

## Outcome

The project now has a second, backward-compatible environment for levels with
the vertical scale and cooperative structure of a full Fireboy & Watergirl
room. It does not alter the earlier fixed-level or lightweight procedural
checkpoints.

The geometry expert solved all 100 generated evaluation seeds from
`100000–100099`:

| Layouts | Successes | Mean length | Mean simultaneous-active frames |
| ---: | ---: | ---: | ---: |
| 100 | 100 | 966.1 | 471.8 |

The machine-readable per-seed results are in
[`temple_expert_100.json`](temple_expert_100.json).

## Mechanics

- Five floors connected by four independently timed moving lifts
- Eleven pools: lava, water, and green acid that kills both characters
- Three red and three blue collectible gems
- Three pairs of character-specific floor switches
- Three gates that open only after both matching switches activate
- Doors that unlock only after all gates and that character's gems are complete
- Solid top and underside collision for stone ledges and moving lifts
- A 1,800-frame limit and dense interaction/ascent rewards

The world is `960×720`, compared with the lightweight environment's `800×400`.
The renderer uses original procedural stonework, vines, lighting, gems,
character shapes, animated pools, lifts, gates, doors, and a HUD. No art from
the reference screenshot is copied into the repository.

## Simultaneous action model

Each character has six local commands:

1. idle
2. left
3. right
4. jump
5. left + jump
6. right + jump

Their Cartesian product is encoded as one `Discrete(36)` action, so a policy
can move and jump both characters on every simulation frame. The geometry-only
observation has 250 values and includes both character states plus padded
platform, lift, hazard, gem, switch, gate, and door state. It contains no level
seed or level identity.

## Commands

```bash
# Watch the simultaneous expert solve a fresh held-out temple.
python watch_temple.py

# Play both characters manually.
python watch_temple.py --manual

# Reproduce the held-out expert benchmark.
python temple_expert.py --count 100

# Verify the demonstration + behavior-cloning + PPO pipeline.
python train_temple.py --quick

# Start a full randomized PPO run.
python train_temple.py
```

The current 100/100 result belongs to the closed-loop geometry expert. The PPO
training pipeline is implemented and smoke-tested, but a fully trained temple
neural checkpoint is not claimed in this report.
