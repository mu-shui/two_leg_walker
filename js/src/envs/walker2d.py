from __future__ import annotations

from typing import Callable, Optional

import gymnasium as gym

from .wrappers import RandomFrictionWrapper


def make_walker2d_env(
    env_id: str,
    seed: int,
    render_mode: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
    random_friction: bool = False,
    friction_low: float = 0.7,
    friction_high: float = 1.3,
) -> Callable[[], gym.Env]:
    def _thunk() -> gym.Env:
        env = gym.make(env_id, render_mode=render_mode, max_episode_steps=max_episode_steps)
        if random_friction:
            env = RandomFrictionWrapper(env, low=friction_low, high=friction_high, seed=seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.reset(seed=seed)
        return env

    return _thunk


def make_walker2d(
    env_id: str,
    seed: int,
    render_mode: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
    random_friction: bool = False,
    friction_low: float = 0.7,
    friction_high: float = 1.3,
) -> gym.Env:
    env = gym.make(env_id, render_mode=render_mode, max_episode_steps=max_episode_steps)
    if random_friction:
        env = RandomFrictionWrapper(env, low=friction_low, high=friction_high, seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env.reset(seed=seed)
    return env
