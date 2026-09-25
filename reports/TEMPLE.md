# Temple environment report

`TempleEnv` is a five-floor cooperative level with simultaneous control. One
`Discrete(36)` action sets a command for Fire and a command for Water on the
same frame: idle, left, right, jump, left+jump, or right+jump for each. The
policy never sees the level seed.

## Results on 1,000 test layouts

Test seeds 100000 to 100999. Every learned row is the checkpoint chosen on
validation layouts before any test layout was played.

| Controller | Type | Success | Hazard deaths | Timeouts | Mean frames | Frames with both characters moving |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Weighted cloning + DAgger, egocentric observation | neural policy | **99.1%** | 0.4% | 0.5% | 957.0 | 49.2% (471.2 per episode) |
| Weighted cloning + DAgger, absolute observation | neural policy | 64.2% | 14.3% | 21.5% | 1377.1 | 37.4% (514.8) |
| Weighted cloning only | neural policy | 0.0% | 43.2% | 56.8% | 1045.3 | 7.1% (74.3) |
| DAgger policy + PPO fine-tuning | neural policy | 0.0% | 2.7% | 97.3% | 1760.8 | 14.5% (255.1) |
| Closed-loop geometry expert | scripted expert | 100.0% | 0.0% | 0.0% | 962.2 | 48.8% (469.5) |

Sources: [`temple_imitation_1000.json`](temple_imitation_1000.json),
[`temple_ppo_1000.json`](temple_ppo_1000.json), and
[`temple_expert_1000.json`](temple_expert_1000.json), each with per-layout
results.

"Frames with both characters moving" counts frames where both characters
receive a non-idle command and both change position, as a share of all frames.
Commands that push into a wall and idling on a moving lift do not count. The
older definition, both commands non-idle, gives 471.9 frames per episode for
the policy and 470.5 for the expert.

The trained policy matches the expert's pace: 957.0 frames per solved or
failed episode against 962.2, with both characters in motion on about half of
all frames.

## Training pipeline

All training and selection used seeds below 100000:

| Split | Seeds | Used for |
| --- | --- | --- |
| Training | 0 to 89,999 | expert demonstrations (0 to 99), DAgger rollouts (1000 to 2599), PPO environments |
| Validation | 90,000 to 90,099 | scoring every saved checkpoint and choosing one |
| Test | 100,000 to 100,999 | the table above, evaluated once per controller |

1. **Expert demonstrations.** The closed-loop expert plays 100 training
   layouts, giving 97,579 labeled frames.
2. **Weighted behavior cloning.** A 512 by 512 ReLU MLP is trained for 20
   epochs. Each character jumps on 700 of the 97,579 demonstration frames
   (0.72%), and unweighted cloning learned to walk into the first pool, so a
   frame where the expert tells a character to jump gets 10 extra units of
   loss weight per jumping character.
3. **DAgger.** Each round runs 100 fresh training layouts with a mixture of
   learner and expert control. The expert's control probability falls from 0.5
   to 0 over the rounds, and the last two rounds run the learner alone. A
   stateless version of the expert labels every visited frame. It reads each
   character's floor from its height, and on 200 validation layouts it chose
   the same action as the original expert on all 193,136 frames.
   The policy retrains for 4 epochs on all data collected so far. After 16
   rounds the dataset holds 1,459,589 frames.
4. **Egocentric features.** The absolute observation lists 250 positions and
   sizes in world coordinates. The egocentric observation appends 20 values
   measured from each character: the distance to the nearest pool on its
   walking surface that would kill it, in each direction, and the height and
   horizontal offset of each lift. A jump decision becomes a threshold on one
   input.
5. **Checkpoint choice.** Every stage and every DAgger round was scored on the
   100 validation layouts. Validation success moved sharply between rounds
   (for example 88%, 0%, 85% in rounds 7 to 9), so the rule takes the round
   with the highest validation score. Round 14 scored 100 of 100; the final
   round 16 scored 18.

Expert checks: [`temple_expert_checks.json`](temple_expert_checks.json).
Validation histories:
[`dagger_egocentric_validation.json`](temple_validation/dagger_egocentric_validation.json),
[`dagger_absolute_validation.json`](temple_validation/dagger_absolute_validation.json),
[`ppo_validation.json`](temple_validation/ppo_validation.json).

The absolute-observation run had 12 DAgger rounds against 16 for the
egocentric run. Within their first 12 rounds the best validation scores were
62 and 90 of 100.

## PPO fine-tuning

PPO started from the round 14 policy with 8 subprocess environments, learning
rate 3e-5, clip range 0.1, and target KL 0.02. The first 10 rollouts trained
only the value network so the critic could catch up with the cloned actor.
Validation success fell from 100 of 100 at the start to 0 of 100 after
250,000 steps and stayed at 0 on all 13 checkpoints through 3,000,000 steps.
The selected checkpoint, the 0% checkpoint with the fewest hazards, solves no
test layout and times out on 97.3% of them.

A likely cause is the same imbalance that broke unweighted cloning: the
decisive frames, jumps at pool edges and boarding a moving lift, are a small
fraction of all frames. An update that barely changes the policy on average
can still flip those rare decisions. PPO did not improve on the imitation
policy in this setting, so the reported temple policy is the DAgger
checkpoint.

## Mechanics

- Five floors connected by four independently timed moving lifts
- Eleven pools: lava, water, and green acid that kills both characters
- Three red and three blue collectible gems
- Three pairs of character-specific floor switches
- Three gates that open only after both matching switches activate
- Doors that unlock only after all gates and that character's gems are complete
- Solid top and underside collision for stone ledges and moving lifts
- A 1,800-frame limit and dense interaction and ascent rewards

The world is 960 by 720 pixels. The renderer draws procedural stonework,
vines, gems, characters, pools, lifts, gates, doors, and a HUD. No art from the
commercial game is included.

## Reproduce

```bash
python scripts/train_temple.py --run-name temple_v4 --timesteps 0 \
  --dagger-iterations 16 --output-dir checkpoints/temple_v4
python scripts/train_temple.py --run-name temple_v4ppo \
  --init-model checkpoints/temple_v4/temple_v4_dagger_14.zip \
  --output-dir checkpoints/temple_v4
python scripts/evaluate_temple.py --expert --output reports/temple_expert_1000.json
python scripts/evaluate_temple.py checkpoints/temple_v4/temple_v4_dagger_14.zip
python scripts/watch_temple.py --model checkpoints/temple_v4/temple_v4_dagger_14.zip
```

The earlier 100-layout expert benchmark is kept in
[`archive/temple/temple_expert_100.json`](archive/temple/temple_expert_100.json).
