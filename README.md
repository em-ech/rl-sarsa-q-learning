# reinforcement-learning-sarsa-q-learning

Tabular TD control on Taxi and Cliff Walking — SARSA (on-policy) and Q-Learning (off-policy) — reproducing Sutton & Barto §6.5's classic on-vs-off-policy comparison.

## Context

Coursework for **Reinforcement Learning** at IE School of Science & Technology (MCSBT, Spring 2026), under Dr. Jaume Manero. Deliverable: a single notebook implementing both algorithms from scratch (no `stable-baselines3`), evaluating them on two environments, and answering five conceptual questions.

Environments come from [Gymnasium](https://gymnasium.farama.org/):

- **Taxi-v4** — 500 discrete states (taxi position × passenger × destination), 6 actions, deterministic transitions, reward −1 / step, +20 dropoff, −10 illegal pickup-or-dropoff.
- **CliffWalking-v1** — 4×12 gridworld, 48 states, 4 actions, deterministic, reward −1 / step and −100 / cliff-cell.

> The assignment PDF references `Taxi-v3` and `CliffWalking-v0`. Gymnasium 1.3 removed both; v4 / v1 are renamed equivalents with identical dynamics.

## Deliverable

| File                                                             | Topic                                                         |
| ---------------------------------------------------------------- | ------------------------------------------------------------- |
| `Assignment2_Emily_Echeverria.ipynb`                             | Single notebook — Activities 1–4 + Section 4 written answers. |
| `taxi_sarsa.mp4`, `taxi_qlearning.mp4`                           | Final greedy rollouts on Taxi.                                |
| `cliff_sarsa.mp4`, `cliff_qlearning.mp4`                         | Final greedy rollouts on Cliff Walking.                       |
| `taxi_sarsa_progression.mp4`, `taxi_qlearning_progression.mp4`   | Snapshots at training checkpoints — Taxi.                     |
| `cliff_sarsa_progression.mp4`, `cliff_qlearning_progression.mp4` | Snapshots at training checkpoints — Cliff Walking.            |

## Headline findings

From `Assignment2_Emily_Echeverria.ipynb`:

- **Taxi: both algorithms converge to near-optimal policies; Q-Learning by a hair.** SARSA greedy +7.89 reward / 13.1 steps. Q-Learning greedy +8.14 / 12.9. Random baseline −765. Deterministic transitions mean there's no cliff for off-policy bootstrap to be naïve about.
- **Cliff Walking: textbook on-vs-off-policy split.** SARSA's training reward (−24.6 last-100) beats Q-Learning's (−53.1) by ~28 points, because Q-Learning's max-target ignores that ε-greedy keeps falling off the cliff during training.
- **Q-Learning learns the theoretically optimal policy in Cliff Walking** (13-step cliff-edge route, total −13). SARSA learns a safer 17-step route along the top of the grid — the on-policy backup folds the 10% ε-greedy exploration probability into nearby Q-values.

Comparative table (MC vs TD(0) vs SARSA vs Q-Learning), full discussion, and the five Section-4 answers live in the notebook.

## Stack

`gymnasium`, `numpy`, `matplotlib`, `imageio`, `tqdm`, `session_info`. Tabular Q-arrays — no neural network. Python 3.12 via the `rl-course` conda env.

## Reproducing

```bash
conda activate rl-course
jupyter lab Assignment2_Emily_Echeverria.ipynb
```

Then `Cell → Run All`. Total runtime ~3 minutes on an M-series Mac. The MP4s in this directory are committed so they can be viewed without re-running training.
