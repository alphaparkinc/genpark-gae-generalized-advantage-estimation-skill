# genpark-gae-generalized-advantage-estimation-skill

Agent Skill implementing **Generalized Advantage Estimation (GAE-$\lambda$)** balancing bias and variance across temporal difference steps for Actor-Critic algorithms.

## Architectural Overview
```mermaid
flowchart TD
    Rewards["Rewards r_t"] & Values["State Values V(s_t)"] --> Delta["TD Residuals: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)"]
    Delta --> Recurse["Backward Accumulator: A_t = delta_t + (gamma * lambda) * A_{t+1}"]
    Recurse --> Advantages["Generalized Advantages A^{GAE}"]
    Advantages & Values --> Returns["Empirical Target Value Returns: R_t = A_t + V(s_t)"]
```
