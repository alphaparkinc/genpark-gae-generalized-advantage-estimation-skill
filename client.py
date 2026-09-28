"""Generalized Advantage Estimation (GAE) Engine.
100% Python Standard Library.
"""

class GeneralizedAdvantageEstimator:
    """GAE(gamma, lambda) exponentially-weighted temporal difference advantages."""
    @staticmethod
    def compute_gae(rewards, values, next_value, gamma=0.99, lam=0.95):
        t_max = len(rewards)
        advantages = [0.0] * t_max
        last_gae = 0.0

        for t in reversed(range(t_max)):
            v_next = values[t + 1] if t + 1 < t_max else next_value
            delta = rewards[t] + gamma * v_next - values[t]
            last_gae = delta + gamma * lam * last_gae
            advantages[t] = last_gae

        returns = [adv + v for adv, v in zip(advantages, values)]
        return {"advantages": advantages, "returns": returns}
