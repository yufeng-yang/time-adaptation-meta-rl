"""Cancel discounted step costs and success-time preference at true endings.

Danger/collision/timeout rewards are left unchanged, including their ordinary
gamma discount. Safe successful episodes have discounted return 1 at any length.
"""
import math

import gymnasium as gym


class DiscountCompensatedTerminalRefund(gym.Wrapper):
    def __init__(self, env, gamma=1.):
        super().__init__(env)
        self.gamma = float(gamma)
        if not 0 < self.gamma <= 1:
            raise ValueError("Discount compensation requires 0 < gamma <= 1")
        self._elapsed = 0
        self._discounted_step_cost = 0.
        self._settled = False

    def reset(self, **kwargs):
        self._elapsed = 0
        self._discounted_step_cost = 0.
        self._settled = False
        observation, info = self.env.reset(**kwargs)
        return observation, dict(info, discounted_step_refund=0., goal_discount_compensation=0.)

    def step(self, action):
        if self._settled:
            raise RuntimeError("Episode already settled; reset before stepping again")
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed += 1
        terminal_discount = self.gamma ** (self._elapsed - 1)
        if terminal_discount == 0:
            raise ArithmeticError("Episode too long for discount compensation")
        self._discounted_step_cost += terminal_discount * self.unwrapped.step_penalty
        refund = goal_compensation = 0.
        if terminated or truncated:
            # A terminal payment is itself discounted: divide by its discount
            # to exactly cancel all discounted per-step deductions, terminal included.
            refund = self._discounted_step_cost / terminal_discount
            if info.get("success", False):
                goal_compensation = self.unwrapped._reward() * (1./terminal_discount - 1.)
            adjustment = refund + goal_compensation
            if not math.isfinite(adjustment):
                raise ArithmeticError("Non-finite terminal compensation")
            reward += adjustment
            # Keep the environment's accounting consistent with emitted rewards.
            self.unwrapped.episode_return += adjustment
            self._settled = True
        info = dict(info, discounted_step_refund=refund,
                    goal_discount_compensation=goal_compensation,
                    episode_return=float(self.unwrapped.episode_return))
        return observation, float(reward), terminated, truncated, info
