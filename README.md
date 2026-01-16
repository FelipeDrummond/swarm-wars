# 🐜 Swarm Wars

> Emergent role specialization in adversarial ant colonies via hierarchical multi-agent RL

Two colonies. Shared food. Limited ants. Learn when to grow, when to specialize, and when to fight.

---

## Overview

Swarm Wars is a minimal environment for studying **emergent role differentiation under adversarial pressure**. Two ant colonies compete for shared food resources, and upon collecting food, must decide: spawn a new ant or specialize an existing one?

The core research question: *Does competition induce meaningful specialization that wouldn't emerge in isolation?*

---

## The Game

```
┌────────────────────────────────────────────────────┐
│  🏠 Nest A                                         │
│     🐜 🐜                                          │
│                    🍎                              │
│                          🍎      🍎               │
│              🍎                                    │
│                                                    │
│                              🐜 🐜                 │
│                                         🏠 Nest B │
└────────────────────────────────────────────────────┘
```

**Setup:**
- 2 colonies start with 2 unspecialized ants each
- Food spawns randomly across a shared arena
- First colony to reach **50 food delivered** wins

**On food delivery, choose one:**
| Action | Effect |
|--------|--------|
| **Bank** | Keep food as score |
| **Spawn** | Create new base ant |
| **Specialize** | Upgrade ant to a role |

**Roles:**
| Role | Vision | Speed | Combat | Strategy |
|------|--------|-------|--------|----------|
| 🐜 Base | 5 | 1.0 | ❌ | Jack of all trades |
| 👁️ Spotter | 10 | 0.8 | ❌ | Scout, find food |
| 🏃 Runner | 4 | 1.8 | ❌ | Fast gatherer |
| ⚔️ Soldier | 5 | 1.0 | ✅ | Eliminate enemies |

---

## Architecture

**Hierarchical MARL with Centralized Training, Decentralized Execution (CTDE)**

```
┌─────────────────────────────────────┐
│        HIGH-LEVEL POLICY            │
│  Colony-wide macro decisions        │
│  • Spawn / Specialize / Bank        │
│  • Which ant to upgrade             │
│  Triggered on: food delivery        │
└─────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│        LOW-LEVEL POLICY             │
│  Per-agent micro decisions          │
│  • Movement direction               │
│  • Attack (soldiers only)           │
│  Conditioned on: role embedding     │
└─────────────────────────────────────┘
```

---

## Installation

```bash
git clone https://github.com/yourusername/swarm-wars.git
cd swarm-wars
pip install -e .
```

**Dependencies:**
- Python 3.9+
- NumPy
- PyTorch
- Gymnasium
- Stable-Baselines3 (or CleanRL)
- Matplotlib / Pygame (visualization)
- Weights & Biases (experiment tracking)

---

## Quick Start

```python
from swarm_wars import SwarmWarsEnv

# Create environment
env = SwarmWarsEnv(
    grid_size=32,
    win_score=50,
    food_spawn_rate=20
)

# Random agents
obs, info = env.reset()
for _ in range(1000):
    actions = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(actions)
    if done:
        break

env.render()
```

---

## Project Structure

```
swarm-wars/
├── swarm_wars/
│   ├── __init__.py
│   ├── env.py              # Core environment
│   ├── agents.py           # Agent classes and roles
│   ├── rendering.py        # Visualization
│   └── utils.py
├── configs/
│   └── default.yaml        # Hyperparameters
├── scripts/
│   ├── train.py            # Training loop
│   ├── evaluate.py         # Evaluation
│   └── visualize.py        # Replay and analysis
├── experiments/
│   └── ...                 # Experiment logs
├── tests/
│   └── ...
├── docs/
│   └── design_doc.md       # Full design document
├── README.md
└── pyproject.toml
```

---

## Research Hypotheses

| # | Hypothesis | Test |
|---|------------|------|
| H1 | Adversarial pressure induces specialization | Compare role diversity: adversarial vs. single-colony |
| H2 | Hierarchy outperforms flat policies | Ablate high-level policy |
| H3 | Emergent strategies are interpretable | Cluster and visualize learned behaviors |

---

## Baselines

- **Random**: Random movement and decisions
- **Greedy Forager**: Move to nearest food, always spawn
- **Balanced Heuristic**: Maintain fixed role ratios
- **Flat RL**: Single policy, no hierarchy
- **No Specialization**: Spawning only, no roles

---

## Roadmap

- [x] Design document
- [ ] Environment implementation
- [ ] Single-colony foraging (sanity check)
- [ ] Hierarchical policy architecture
- [ ] Adversarial self-play
- [ ] Baselines and ablations
- [ ] Paper draft

---

## Citation

```bibtex
@misc{swarmwars2025,
  author = {Felipe},
  title = {Swarm Wars: Emergent Role Specialization in Adversarial Swarms},
  year = {2025},
  url = {https://github.com/yourusername/swarm-wars}
}
```

---

## License

MIT

---

## Acknowledgments

Built as a side project exploring hierarchical RL for swarm robotics. Inspired by ant colony optimization, StarCraft micromanagement, and the question: *what emerges when you let agents decide what to become?*