from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np


class RandomFrictionWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        low: float = 0.7,
        high: float = 1.3,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(env)
        if low <= 0 or high <= 0:
            raise ValueError("friction range must be positive")
        if low > high:
            raise ValueError("friction low must be <= high")
        self.low = float(low)
        self.high = float(high)
        self._rng = np.random.default_rng(seed)
        self._base_friction: Optional[np.ndarray] = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._apply_random_friction()
        return obs, info

    def _apply_random_friction(self) -> Optional[float]:
        model = getattr(self.env.unwrapped, "model", None)
        if model is None or not hasattr(model, "geom_friction"):
            return None
        if self._base_friction is None:
            self._base_friction = model.geom_friction.copy()
        scale = float(self._rng.uniform(self.low, self.high))
        model.geom_friction[:] = self._base_friction * scale
        return scale
