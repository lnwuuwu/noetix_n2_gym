"""Adversarial Motion Prior discriminator.

The AMP reward is derived from a least-squares discriminator with expert label
``+1`` and policy label ``-1``.  Using a WGAN objective here (as the previous
implementation did) is mathematically inconsistent with that reward and gives
the policy an arbitrary signal.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class AMPDiscriminator(nn.Module):
    def __init__(
        self,
        input_dim: int = 110,
        hidden_dims: Sequence[int] = (1024, 512),
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("AMP discriminator input_dim must be positive")
        if not hidden_dims or any(int(width) <= 0 for width in hidden_dims):
            raise ValueError("AMP discriminator hidden_dims must be positive")

        layers = []
        current_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            linear = nn.Linear(current_dim, int(hidden_dim))
            nn.init.orthogonal_(linear.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(linear.bias)
            layers.extend((linear, nn.ReLU()))
            current_dim = int(hidden_dim)

        output = nn.Linear(current_dim, 1)
        nn.init.uniform_(output.weight, -0.05, 0.05)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.net = nn.Sequential(*layers)

    def forward(self, transitions: torch.Tensor) -> torch.Tensor:
        if transitions.ndim != 2:
            raise ValueError(
                "AMP discriminator expects [batch, features], received "
                + str(tuple(transitions.shape))
            )
        return self.net(transitions)

    def compute_style_reward_from_transition(
        self, transitions: torch.Tensor
    ) -> torch.Tensor:
        """Return the bounded AMP reward from normalized transitions."""
        discriminator_value = self.forward(transitions)
        reward = torch.clamp(
            1.0 - 0.25 * torch.square(discriminator_value - 1.0),
            min=0.0,
            max=1.0,
        )
        return reward.squeeze(-1)

    def compute_style_reward(
        self, state: torch.Tensor, next_state: torch.Tensor
    ) -> torch.Tensor:
        return self.compute_style_reward_from_transition(
            torch.cat((state, next_state), dim=-1)
        )

    def compute_gradient_penalty(
        self,
        expert_data: torch.Tensor,
        coefficient: float = 10.0,
    ) -> torch.Tensor:
        """Penalize the discriminator gradient on the expert manifold."""
        if coefficient < 0.0:
            raise ValueError("Gradient-penalty coefficient cannot be negative")
        expert_data = expert_data.detach().requires_grad_(True)
        expert_output = self.forward(expert_data)
        gradients = torch.autograd.grad(
            outputs=expert_output,
            inputs=expert_data,
            grad_outputs=torch.ones_like(expert_output),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return coefficient * torch.mean(
            torch.sum(torch.square(gradients), dim=-1)
        )

    def compute_loss(
        self,
        expert_data: torch.Tensor,
        policy_data: torch.Tensor,
        gradient_penalty_coefficient: float = 10.0,
    ):
        """Return the least-squares AMP loss and detached diagnostics."""
        if expert_data.shape != policy_data.shape:
            raise ValueError(
                "Expert and policy AMP batches must match exactly: {} != {}".format(
                    tuple(expert_data.shape), tuple(policy_data.shape)
                )
            )
        expert_output = self.forward(expert_data)
        policy_output = self.forward(policy_data)
        expert_loss = 0.5 * torch.mean(torch.square(expert_output - 1.0))
        policy_loss = 0.5 * torch.mean(torch.square(policy_output + 1.0))
        gradient_penalty = self.compute_gradient_penalty(
            expert_data, coefficient=gradient_penalty_coefficient
        )
        total_loss = expert_loss + policy_loss + gradient_penalty
        diagnostics = {
            "expert_loss": expert_loss.detach(),
            "policy_loss": policy_loss.detach(),
            "gradient_penalty": gradient_penalty.detach(),
            "expert_score": expert_output.detach().mean(),
            "policy_score": policy_output.detach().mean(),
            "expert_accuracy": (expert_output.detach() > 0.0).float().mean(),
            "policy_accuracy": (policy_output.detach() < 0.0).float().mean(),
        }
        return total_loss, diagnostics

    # Backward-compatible name used by early local AMP experiments.
    def compute_grad_pen(self, real_data, fake_data=None, lambda_gp=10.0):
        del fake_data
        return self.compute_gradient_penalty(real_data, lambda_gp)
