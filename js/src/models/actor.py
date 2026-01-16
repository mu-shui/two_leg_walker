import torch
import torch.nn as nn


def build_mlp(input_dim: int, hidden_sizes: list[int], output_dim: int, activation=nn.ReLU) -> nn.Sequential:
    layers = []
    prev = input_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(prev, size))
        layers.append(activation())
        prev = size
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class ActorMLP(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_limit: float,
        hidden_sizes: list[int],
    ) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_sizes, action_dim)
        self.action_limit = float(action_limit)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        action = torch.tanh(self.net(obs))
        return action * self.action_limit
