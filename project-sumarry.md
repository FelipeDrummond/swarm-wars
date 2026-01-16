# Adversarial Swarm Specialization
## Emergent Role Differentiation Under Competitive Pressure

**Working Title:** *Learning to Grow and Specialize: Hierarchical Multi-Agent RL in Adversarial Foraging*

---

## 1. Problem Statement

Two ant colonies compete for shared food resources. Each colony starts with identical, unspecialized agents. Upon collecting food, a colony must decide: **grow** (spawn new ant) or **specialize** (upgrade existing ant into a role). 

**Core question:** Does adversarial pressure induce meaningful role specialization and strategic population management through hierarchical reinforcement learning?

---

## 2. Environment Specification

### 2.1 World

| Property | Value |
|----------|-------|
| Space | 2D continuous (or discrete grid, e.g., 32×32) |
| Boundary | Walled arena |
| Nests | 2, placed at opposite corners |
| Food sources | Spawn randomly, respawn after depletion |
| Fog of war | Agents only see within their vision radius |

### 2.2 Agents

**Base ant (unspecialized):**
| Attribute | Value |
|-----------|-------|
| Vision radius | 5 units |
| Speed | 1.0 |
| Can attack | No |
| Carry capacity | 1 food |

**Specialized classes:**

| Class | Vision | Speed | Combat | Tradeoff |
|-------|--------|-------|--------|----------|
| **Spotter** | 10 | 0.8 | No | Sees far, slow, fragile |
| **Runner** | 4 | 1.8 | No | Fast gatherer, limited vision |
| **Soldier** | 5 | 1.0 | Yes (kills in 1 hit) | Can't carry food |

### 2.3 State Space

**Global state (for centralized critic):**
- Positions and types of all agents (both colonies)
- Food source positions
- Colony food banks (accumulated score)
- Time step

**Local observation (per agent):**
- Relative positions of visible allies, enemies, food, nest
- Own type (one-hot: base, spotter, runner, soldier)
- Own colony's food bank (normalized)
- Could include: local pheromone signals (optional, v2+)

### 2.4 Action Spaces

**Low-level (per agent, continuous or discrete):**
- Movement direction (2D vector or 4/8 discrete directions)
- If soldier: attack toggle (when enemy in range)

*Note: No explicit "return to nest" action. The agent must learn that delivering food yields reward and navigate back on its own.*

**High-level (per colony, triggered on food delivery):**
```
When food delivered to nest:
  → Choose one of:
      - BANK: Keep food as score (no spawn/upgrade)
      - SPAWN: Create new base ant at nest
      - SPECIALIZE(agent_id, class): Upgrade target ant
```

### 2.5 Reward Function

**Colony-level reward (shared by all agents in colony):**

```python
reward = 0

# Food collection (main objective)
reward += food_delivered * 1.0

# Survival bonus (small)
reward += alive_agents * 0.01

# Enemy kills (optional, can shape aggression)
reward += enemies_killed * 0.2

# Penalty for agent death
reward -= own_deaths * 0.3

# Terminal bonus
if episode_ends:
    if own_food > enemy_food:
        reward += 10.0  # Win bonus
    elif own_food < enemy_food:
        reward -= 5.0   # Lose penalty
```

**Shaping rewards (for early training, anneal over time):**
- +0.01 for exploring new areas
- +0.05 for spotting food (spotter bonus)
- +0.1 for picking up food

### 2.6 Episode Structure

| Parameter | Value |
|-----------|-------|
| Win score | 50 food delivered (tunable) |
| Max steps | 2000 (safety cap, should rarely hit) |
| Starting ants per colony | 2 (both base) |
| Food spawn rate | 1 per 20 steps (tunable) |
| Termination | First colony to win score OR max steps OR one colony eliminated |

*Note: Score-based termination prevents runaway agent counts in long games and creates clear win conditions.*

---

## 3. Hierarchy Definition

### 3.1 Two-Level Architecture

```
┌─────────────────────────────────────────┐
│           HIGH-LEVEL POLICY             │
│  (Colony-wide, triggered on events)     │
│                                         │
│  Input: Colony state summary            │
│  Output: Macro decision                 │
│    - Spawn / Specialize / Bank          │
│    - Target agent for specialization    │
│    - Target class                       │
│                                         │
│  Temporal scale: ~50-100 steps          │
└─────────────────────────────────────────┘
                    │
                    │ Role assignments + spawn decisions
                    ▼
┌─────────────────────────────────────────┐
│           LOW-LEVEL POLICY              │
│  (Per-agent, every step)                │
│                                         │
│  Input: Local observation + own role    │
│  Output: Movement + action              │
│                                         │
│  Temporal scale: Every step             │
│                                         │
│  Architecture: Shared weights across    │
│  agents, conditioned on role embedding  │
└─────────────────────────────────────────┘
```

### 3.2 High-Level Policy Details

**Input features:**
- Own colony: [n_base, n_spotter, n_runner, n_soldier, food_bank]
- Enemy colony: [n_total_visible, n_soldiers_visible] (partial info)
- Game state: [time_remaining_normalized, food_on_map]

**Output:** Discrete action space
```
Actions:
  0: BANK
  1: SPAWN_BASE
  2: SPECIALIZE → SPOTTER (requires selecting agent)
  3: SPECIALIZE → RUNNER
  4: SPECIALIZE → SOLDIER
```

If specialization chosen, second head selects which base ant to upgrade.

### 3.3 Low-Level Policy Details

**Architecture:** Parameter-shared policy with role conditioning

```python
class LowLevelPolicy(nn.Module):
    def __init__(self):
        self.obs_encoder = MLP(obs_dim, 64)
        self.role_embed = nn.Embedding(4, 16)  # base, spotter, runner, soldier
        self.policy_head = MLP(64 + 16, action_dim)
    
    def forward(self, obs, role):
        obs_feat = self.obs_encoder(obs)
        role_feat = self.role_embed(role)
        return self.policy_head(torch.cat([obs_feat, role_feat], dim=-1))
```

*Future work: Graph Neural Network over agents (message passing) with role as node feature. Deferred to v2 to keep initial scope tractable.*

---

## 4. Training Setup

### 4.1 Algorithm

**Framework:** Centralized Training, Decentralized Execution (CTDE)

- **Policy optimization:** PPO (stable, works well for multi-agent)
- **Critic:** Centralized, sees full state (both colonies)
- **Self-play:** Train against past versions of self

### 4.2 Self-Play Curriculum

```
Phase 1: vs Random policy          (learn basic foraging)
Phase 2: vs Frozen snapshot        (learn to beat fixed strategy)  
Phase 3: vs Self (online)          (co-evolution)
Phase 4: vs Population pool        (robustness)
```

### 4.3 Hyperparameters (Starting Point)

| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-4 |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| PPO clip | 0.2 |
| Entropy bonus | 0.01 (anneal to 0.001) |
| Batch size | 2048 steps |
| N parallel envs | 16-32 |

---

## 5. Hypotheses

### H1: Adversarial pressure induces specialization
**Prediction:** Colonies trained against adversaries will develop heterogeneous populations, while colonies trained in isolation (single-colony foraging) will remain homogeneous or show degenerate specialization.

**Test:** Compare population entropy (diversity of roles) between:
- Adversarial training
- Single-colony training (no enemy)
- Fixed-strategy opponent (no adaptation)

### H2: Hierarchical architecture outperforms flat
**Prediction:** Separating macro-decisions (spawn/specialize) from micro-decisions (movement) improves sample efficiency and final performance.

**Test:** Compare against baselines:
- Flat policy (single network decides everything)
- Fixed heuristic high-level (e.g., always spawn until N, then specialize evenly)
- Random high-level, learned low-level

### H3: Emergent strategies exhibit interpretable structure
**Prediction:** Learned policies will converge to recognizable strategies (e.g., "rush soldiers early", "scout then expand", "runner spam") that can be categorized and analyzed.

**Test:** 
- Cluster final population compositions across many episodes
- Visualize decision boundaries of high-level policy
- Compare against hand-designed strategies

---

## 6. Baselines

| Baseline | Description |
|----------|-------------|
| **Random** | Random movement, random high-level decisions |
| **Greedy forager** | Move toward nearest food, always spawn |
| **Balanced heuristic** | Maintain 1:1:1 ratio of spotter:runner:soldier |
| **Flat RL** | Single policy, no hierarchy |
| **No specialization** | Only spawn allowed, no role upgrades |
| **Single colony** | Same task, no adversary |

---

## 7. Metrics & Analysis

### 7.1 Performance Metrics
- Win rate (against fixed opponents and self-play)
- Food collected per episode
- Survival rate (% agents alive at end)

### 7.2 Emergence Metrics
- Population composition over time (area chart of role counts)
- Role diversity (entropy of population distribution)
- Strategy clustering (t-SNE of high-level decision trajectories)

### 7.3 Ablation Studies
- Remove each role class → which is essential?
- Remove high-level → can flat policy match?
- Vary food scarcity → does strategy shift?
- Vary starting ants → robustness check

---

## 8. Implementation Roadmap

### Phase 0: Environment (Week 1-2)
- [ ] Implement base 2D world with movement physics
- [ ] Add food spawning and collection
- [ ] Add nests and food delivery
- [ ] Add vision/fog of war
- [ ] Implement agent classes with different stats
- [ ] Add combat mechanics
- [ ] Render with matplotlib or pygame

### Phase 1: Single Colony Foraging (Week 3)
- [ ] Train low-level policy only (no spawning, fixed 2 agents)
- [ ] Verify agents learn to find and return food
- [ ] Add spawning, verify population growth works

### Phase 2: Hierarchy + Specialization (Week 4-5)
- [ ] Implement high-level policy
- [ ] Train with specialization enabled
- [ ] Verify roles are being assigned
- [ ] Check for degenerate solutions

### Phase 3: Adversarial (Week 6-7)
- [ ] Add second colony
- [ ] Implement self-play training loop
- [ ] Train and iterate on reward shaping
- [ ] Analyze emergent strategies

### Phase 4: Experiments & Writing (Week 8-10)
- [ ] Run all baselines
- [ ] Run ablations
- [ ] Generate visualizations
- [ ] Write paper draft

---

## 9. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| One role dominates (degenerate) | Medium | Rebalance class stats; add counters |
| Self-play collapse (both do nothing) | Medium | Add exploration bonus; population training |
| Hierarchy doesn't help | Low | This itself is a finding; analyze why |
| Training instability | Medium | Careful reward shaping; curriculum |
| Simulation too slow | Low | Vectorize env; use JAX if needed |

---

## 10. Open Questions

1. Should food have types (requiring different specialists)?
2. Should there be a cost to specialization (not just opportunity cost)?
3. Should communication be explicit (signals) or implicit (observation only)?
4. Should we allow "de-specialization" or is it permanent?
5. What map structure encourages interesting strategies?
6. **(Future)** Does GNN architecture improve coordination vs. independent MLPs?

---

## Appendix: Tech Stack

| Component | Choice |
|-----------|--------|
| Simulation | Python + NumPy (start), JAX (if needed) |
| RL Framework | Stable-Baselines3 or CleanRL |
| Experiment tracking | Weights & Biases |
| Visualization | Matplotlib, Pygame for replay |
| Compute | Local GPU or Colab (prototyping) |

---

*Last updated: January 2025*
*Author: Felipe*