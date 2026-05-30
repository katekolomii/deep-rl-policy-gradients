# Deep RL Policy Gradients
**COSC 3440 Deep Reinforcement Learning | Georgetown University, Spring 2026**

Implementation and comparison of three policy gradient algorithms on the **Ant-v5** continuous control environment (MuJoCo). The agent must coordinate 8 joints to walk forward, with a reward signal combining forward velocity, a survival bonus, and a control-cost penalty.

---

## Algorithms

1. **REINFORCE** — vanilla policy gradient
2. **PG with Baseline** — policy gradient with a value function baseline
3. **PPO** — Proximal Policy Optimization with clipped surrogate objective and GAE

---

## Setup

Python 3.12+

```bash
pip install gymnasium[mujoco] mujoco torch numpy matplotlib
```

Run all cells top to bottom in the notebook (Kernel → Restart & Run All).

**Expected runtime: ~2–4 hours (CPU). No GPU required.**

---

## Hyperparameters

**Shared:**
| Parameter | Value |
|---|---|
| gamma (discount) | 0.99 |
| episodes_per_batch | 10 |
| total_episodes | 3000 |
| max_steps_per_episode | 1000 |
| hidden_sizes | (128, 128) |
| log_std initialization | -1.5 |

**Per algorithm:**
| Algorithm | policy_lr | value_lr | other |
|---|---|---|---|
| PG | 1e-4 | — | — |
| PGB | 1e-4 | 1e-3 | — |
| PPO | 3e-4 | 1e-3 | clip_eps=0.2, gae_lambda=0.95, policy_epochs=10, value_epochs=10, minibatch_size=256 |

**Q3 sweep (policy_lr):** `1e-4`, `1e-3`, `3e-3` (Q2 baseline: `3e-4`)

---

## Results

### Algorithm Comparison (Q2)

| Algorithm | Start (ep. 0) | End (ep. 2000) |
|---|---|---|
| REINFORCE | 175 | 276 |
| PG + Baseline | ~175 | ~280 |
| PPO | 241 | 1565 |

PPO dramatically outperformed both baselines, reaching a return of 1000 by episode 750 while REINFORCE and PG with Baseline never exceeded 350. With only 10 trajectories per update, the value baseline lacked sufficient data to meaningfully stabilize gradient estimates.

### PPO Learning Rate Sweep (Q3)

| Learning Rate | Behavior |
|---|---|
| `1e-4` | Best long-term — reached ~1500 by ep. 2000, still trending upward |
| `1e-3` | Fast initially (~700 by ep. 500) but plateaued at 750–800; overtaken by `1e-4` around ep. 1000 |
| `3e-3` | Collapsed — degraded to negative returns by ep. 750 |

Conservative learning rates are preferable for complex continuous control: too high a learning rate causes destructive policy updates that PPO's clipping cannot fully prevent.

---

## Key Implementation Details

- Gaussian policy with state-independent log_std
- Observation normalization (running mean/std)
- Reward-to-go returns for PG and PGB
- GAE (Generalized Advantage Estimation) for PPO
- Gradient clipping (max_norm=0.5) for PG and PGB
- Minibatch updates for PPO
