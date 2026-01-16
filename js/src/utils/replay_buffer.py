import random
from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, size: int, device: torch.device) -> None:
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((size,), dtype=np.float32)
        self.dones = np.zeros((size,), dtype=np.float32)
        self.max_size = int(size)
        self.ptr = 0
        self.size = 0
        self.device = device

    def add(self, obs, action, reward, next_obs, done) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def ready(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=torch.as_tensor(self.obs[idx], device=self.device),
            actions=torch.as_tensor(self.actions[idx], device=self.device),
            rewards=torch.as_tensor(self.rewards[idx], device=self.device),
            next_obs=torch.as_tensor(self.next_obs[idx], device=self.device),
            dones=torch.as_tensor(self.dones[idx], device=self.device),
        )
        return batch


class SequenceReplayBuffer:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_transitions: int,
        seq_len: int,
        device: torch.device,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_transitions = int(max_transitions)
        self.seq_len = int(seq_len)
        self.device = device
        self.episodes: Deque[List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]] = deque()
        self.current: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = []
        self.size = 0

    def add(self, obs, action, reward, next_obs, done) -> None:
        transition = (
            np.asarray(obs, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            np.asarray(next_obs, dtype=np.float32),
            bool(done),
        )
        self.current.append(transition)
        self.size += 1
        if done:
            self._flush_current()
        self._trim()

    def _flush_current(self) -> None:
        if self.current:
            self.episodes.append(self.current)
            self.current = []

    def _trim(self) -> None:
        while self.size > self.max_transitions and self.episodes:
            ep = self.episodes.popleft()
            self.size -= len(ep)

    def ready(self, batch_size: int) -> bool:
        return self._num_candidates() >= batch_size

    def _num_candidates(self) -> int:
        total = 0
        for ep in self._candidate_episodes():
            total += max(0, len(ep) - self.seq_len + 1)
        return total

    def _candidate_episodes(self):
        episodes = list(self.episodes)
        if len(self.current) >= self.seq_len:
            episodes.append(self.current)
        return episodes

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        episodes = self._candidate_episodes()
        if not episodes:
            raise RuntimeError("Sequence buffer is empty")

        obs_seq = np.zeros((batch_size, self.seq_len, self.obs_dim), dtype=np.float32)
        next_obs_seq = np.zeros((batch_size, self.seq_len, self.obs_dim), dtype=np.float32)
        actions = np.zeros((batch_size, self.action_dim), dtype=np.float32)
        rewards = np.zeros((batch_size,), dtype=np.float32)
        dones = np.zeros((batch_size,), dtype=np.float32)

        for i in range(batch_size):
            ep = random.choice(episodes)
            if len(ep) < self.seq_len:
                ep = random.choice([e for e in episodes if len(e) >= self.seq_len])
            start = random.randint(0, len(ep) - self.seq_len)
            for j in range(self.seq_len):
                obs_seq[i, j] = ep[start + j][0]
            last = ep[start + self.seq_len - 1]
            actions[i] = last[1]
            rewards[i] = last[2]
            dones[i] = float(last[4])
            for j in range(self.seq_len - 1):
                next_obs_seq[i, j] = ep[start + j + 1][0]
            next_obs_seq[i, -1] = last[3]

        batch = dict(
            obs_seq=torch.as_tensor(obs_seq, device=self.device),
            actions=torch.as_tensor(actions, device=self.device),
            rewards=torch.as_tensor(rewards, device=self.device),
            next_obs_seq=torch.as_tensor(next_obs_seq, device=self.device),
            dones=torch.as_tensor(dones, device=self.device),
        )
        return batch
