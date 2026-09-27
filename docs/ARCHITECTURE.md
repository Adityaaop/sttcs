# Technical Architecture Specification

## 1. Environment and State Space Formulation

The railway corridor is modeled as a directed graph $G = (V, E)$, where $V$ represents track segments and station loops, and $E$ represents valid traversal connections.

The state vector at time $t$, $S_t$, is comprised of:
- **Topology Features**: For each node $v \in V$, we capture its static capacity $C_v$, length $L_v$, and disruption status $D_v \in \{0, 1\}$.
- **Dynamic Occupancy**: Number of trains currently within the node boundaries.
- **Train State**: For train $i$, its absolute position $P_i$, velocity $V_i$, direction $d_i$, and priority class $w_i$.

To handle variable numbers of trains and track segments, a **Graph Neural Network (GNN)** encoder compresses this highly dynamic state into a fixed-length 128-dimensional embedding vector $z_t$.

## 2. Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

We define a set of $N$ Section Agents, each responsible for a distinct geographical sector. 

### Action Space ($A_t$)
For each train $i$ inside an agent's sector, the agent outputs:
- $a_{v, i} \in [0, 160]$ (Continuous Speed Advisory)
- $a_{s, i} \in \{0, 1, 2\}$ (Discrete Signal Aspect: Red, Yellow, Green)

### Multi-Objective Reward Function
The reward $R_t$ at each timestep is globally shared and composed of four competing objectives:

$$ R_t = \alpha \cdot R_{\text{throughput}} - \beta \cdot R_{\text{delay}} + \gamma \cdot R_{\text{safety\_margin}} + \delta \cdot R_{\text{util}} $$

Where:
- $R_{\text{throughput}}$ = $\sum (V_i / V_{max})$
- $R_{\text{delay}}$ = $\sum \max(0, ETA_i - Deadline_i) \cdot w_i$
- $R_{\text{safety\_margin}}$ = $\min(\text{Distance}(i, j) - 2.0, 0)$
- Weights: $\alpha=0.3, \beta=0.3, \gamma=0.2, \delta=0.2$

### Centralized Critic
During training, the centralized critic $Q^\pi(z_t, A_1, A_2, ..., A_N)$ evaluates the joint actions of all section controllers, allowing decentralized execution at inference time while learning cooperative behaviors (e.g., yielding right-of-way).

## 3. Level 1 Failsafe: Mathematical Proofs

The deterministic wrapper operates continuously to intercept unsafe AI outputs.

**Invariant 1: Absolute Headway Separation**
Let $P_1, P_2$ be the positions of two consecutive trains on the same track edge, moving at speeds $V_1, V_2$. Let $H_{min} = 2.0 \text{ km}$.
The failsafe ensures that at $t+1$:
$$ (P_1 + V_1 \cdot \Delta t) - (P_2 + V_2 \cdot \Delta t) \geq H_{min} $$
If the AI-advised $V_2^*$ violates this inequality, the failsafe solves for $V_2$ and truncates the action:
$$ V_2 = \min \left( V_2^*, \frac{(P_1 - P_2 - H_{min})}{\Delta t} + V_1 \right) $$

**Invariant 2: SPAD (Signal Passed At Danger) Avoidance**
If the block ahead has signal aspect $S = 0$ (Red), and distance to signal is $D_{sig}$:
$$ \text{If } D_{sig} \leq (V_i^2 / (2 \cdot a_{braking})), \text{ force } V_i = 0 $$

## 4. API Schemas

### `/simulate/step` (POST)
Ingests telemetry actions and advances the simulation.
```json
{
  "actions": {
    "T_Rajdhani_01": {
      "speed_advisory": 100.0,
      "signal_state": 2
    }
  }
}
```

### `/disruption/inject` (POST)
Injects scenarios for What-If evaluation.
```json
{
  "type": "track_blockage",
  "location": "Station_1",
  "duration_steps": 100,
  "severity": 1.0
}
```
