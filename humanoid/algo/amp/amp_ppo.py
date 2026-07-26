"""PPO with an Adversarial Motion Prior auxiliary reward."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.optim as optim

from humanoid.algo.amp.amp_storage import AMPRolloutStorage
from humanoid.algo.amp.observations import (
    AMP_OBSERVATION_DIM,
    mask_unsupported_amp_features,
)
from humanoid.algo.ppo.ppo import PPO


class AMPPPO(PPO):
    """Add a normalized least-squares AMP discriminator to standard PPO."""

    def __init__(
        self,
        policy,
        discriminator,
        expert_observation_mean,
        expert_observation_std,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg: Optional[dict] = None,
        multi_gpu_cfg: Optional[dict] = None,
        amp_replay_buffer_size=100000,
        amp_batch_size=512,
        style_reward_weight=0.2,
        disc_learning_rate=1e-4,
        disc_updates_per_iteration=2,
        gradient_penalty_coefficient=10.0,
        reward_warmup_updates=10,
        reward_ramp_updates=100,
    ):
        super().__init__(
            policy=policy,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            device=device,
            normalize_advantage_per_mini_batch=(
                normalize_advantage_per_mini_batch
            ),
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        self.discriminator = discriminator.to(device)
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=float(disc_learning_rate)
        )

        self.style_reward_weight = self._finite_non_negative(
            "style_reward_weight", style_reward_weight
        )
        self.amp_batch_size = self._positive_int(
            "amp_batch_size", amp_batch_size
        )
        self.amp_replay_buffer_size = self._positive_int(
            "amp_replay_buffer_size", amp_replay_buffer_size
        )
        self.disc_updates_per_iteration = self._positive_int(
            "disc_updates_per_iteration", disc_updates_per_iteration
        )
        self.gradient_penalty_coefficient = self._finite_non_negative(
            "gradient_penalty_coefficient",
            gradient_penalty_coefficient,
        )
        self.reward_warmup_updates = self._non_negative_int(
            "reward_warmup_updates", reward_warmup_updates
        )
        self.reward_ramp_updates = self._non_negative_int(
            "reward_ramp_updates", reward_ramp_updates
        )

        mean = torch.as_tensor(
            expert_observation_mean, dtype=torch.float32, device=self.device
        ).reshape(-1)
        std = torch.as_tensor(
            expert_observation_std, dtype=torch.float32, device=self.device
        ).reshape(-1)
        if mean.numel() != AMP_OBSERVATION_DIM or std.numel() != AMP_OBSERVATION_DIM:
            raise ValueError(
                "Expert AMP statistics must each have {} entries".format(
                    AMP_OBSERVATION_DIM
                )
            )
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("Expert AMP statistics contain NaN or Inf")
        self.expert_observation_mean = mean
        self.expert_observation_std = torch.clamp(std, min=1.0e-4)

        self.amp_replay_buffer = None
        self.amp_replay_ptr = 0
        self.amp_replay_full = False
        self.motion_loader = None
        self.discriminator_updates_completed = 0
        self._style_reward_sum = 0.0
        self._style_reward_count = 0
        self._effective_style_weight_sum = 0.0

    @staticmethod
    def _finite_non_negative(name, value):
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("{} must be finite and non-negative".format(name))
        return value

    @staticmethod
    def _positive_int(name, value):
        value = int(value)
        if value < 1:
            raise ValueError("{} must be positive".format(name))
        return value

    @staticmethod
    def _non_negative_int(name, value):
        value = int(value)
        if value < 0:
            raise ValueError("{} cannot be negative".format(name))
        return value

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
    ):
        self.storage = AMPRolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            device=self.device,
        )

    def _normalize_observation(self, observation):
        observation = mask_unsupported_amp_features(observation)
        return torch.clamp(
            (
                observation - self.expert_observation_mean
            )
            / self.expert_observation_std,
            min=-10.0,
            max=10.0,
        )

    def _normalized_transition(self, state, next_state):
        if state.shape != next_state.shape:
            raise ValueError(
                "AMP state pair shapes differ: {} != {}".format(
                    tuple(state.shape), tuple(next_state.shape)
                )
            )
        if state.ndim != 2 or state.shape[1] != AMP_OBSERVATION_DIM:
            raise ValueError(
                "AMP state must be [batch, {}], received {}".format(
                    AMP_OBSERVATION_DIM, tuple(state.shape)
                )
            )
        if not torch.isfinite(state).all() or not torch.isfinite(next_state).all():
            raise FloatingPointError("AMP transition contains NaN or Inf")
        return torch.cat(
            (
                self._normalize_observation(state),
                self._normalize_observation(next_state),
            ),
            dim=-1,
        )

    @property
    def effective_style_reward_weight(self):
        completed = int(self.discriminator_updates_completed)
        if completed < self.reward_warmup_updates:
            return 0.0
        if self.reward_ramp_updates == 0:
            return self.style_reward_weight
        progress = min(
            1.0,
            (completed - self.reward_warmup_updates + 1)
            / float(self.reward_ramp_updates),
        )
        return self.style_reward_weight * progress

    def process_env_step(self, rewards, dones, infos):
        state = getattr(self.transition, "amp_obs", None)
        next_state = getattr(self.transition, "amp_obs_next", None)
        if state is None or next_state is None:
            raise RuntimeError(
                "AMP observations were not attached before process_env_step"
            )
        transitions = self._normalized_transition(state, next_state)
        valid = ~dones.reshape(-1).bool()

        with torch.no_grad():
            style_reward = (
                self.discriminator.compute_style_reward_from_transition(
                    transitions
                )
            )
            style_reward = torch.where(
                valid, style_reward, torch.zeros_like(style_reward)
            )

        effective_weight = self.effective_style_reward_weight
        combined_rewards = rewards + effective_weight * style_reward
        self._update_replay_buffer(transitions, valid_mask=valid)

        valid_count = int(valid.sum().item())
        if valid_count:
            self._style_reward_sum += float(style_reward[valid].sum().item())
            self._style_reward_count += valid_count
            self._effective_style_weight_sum += (
                effective_weight * valid_count
            )
        super().process_env_step(combined_rewards, dones, infos)

    def _update_replay_buffer(self, transitions, valid_mask=None):
        """Append normalized transitions to a bounded circular buffer."""
        transitions = transitions.detach()
        if valid_mask is not None:
            transitions = transitions[valid_mask.reshape(-1).bool()]
        if transitions.numel() == 0:
            return
        if transitions.ndim != 2:
            raise ValueError("AMP replay data must be two-dimensional")

        pair_dim = transitions.shape[1]
        if self.amp_replay_buffer is None:
            self.amp_replay_buffer = torch.empty(
                self.amp_replay_buffer_size,
                pair_dim,
                dtype=transitions.dtype,
                device=self.device,
            )
        elif self.amp_replay_buffer.shape[1] != pair_dim:
            raise ValueError("AMP replay transition width changed")

        count = transitions.shape[0]
        if count >= self.amp_replay_buffer_size:
            self.amp_replay_buffer.copy_(
                transitions[-self.amp_replay_buffer_size :]
            )
            self.amp_replay_ptr = 0
            self.amp_replay_full = True
            return

        end = self.amp_replay_ptr + count
        if end <= self.amp_replay_buffer_size:
            self.amp_replay_buffer[self.amp_replay_ptr : end].copy_(
                transitions
            )
        else:
            first_count = self.amp_replay_buffer_size - self.amp_replay_ptr
            self.amp_replay_buffer[self.amp_replay_ptr :].copy_(
                transitions[:first_count]
            )
            self.amp_replay_buffer[: end - self.amp_replay_buffer_size].copy_(
                transitions[first_count:]
            )
            self.amp_replay_full = True
        self.amp_replay_ptr = end % self.amp_replay_buffer_size
        if end >= self.amp_replay_buffer_size:
            self.amp_replay_full = True

    @property
    def replay_size(self):
        if self.amp_replay_buffer is None:
            return 0
        return (
            self.amp_replay_buffer_size
            if self.amp_replay_full
            else self.amp_replay_ptr
        )

    def _sample_expert_transitions(self, sample_size):
        generator = self.motion_loader.feed_forward_generator(1, sample_size)
        expert_states, _labels = next(generator)
        if expert_states.ndim != 3 or expert_states.shape[1] != 2:
            raise ValueError(
                "Expert AMP states must be [batch, 2, features], received "
                + str(tuple(expert_states.shape))
            )
        if expert_states.shape[2] != AMP_OBSERVATION_DIM:
            raise ValueError(
                "Expert AMP width is {}, expected {}".format(
                    expert_states.shape[2], AMP_OBSERVATION_DIM
                )
            )
        return torch.cat(
            (
                self._normalize_observation(expert_states[:, 0]),
                self._normalize_observation(expert_states[:, 1]),
            ),
            dim=-1,
        )

    def _update_discriminator(self):
        if self.motion_loader is None or self.replay_size < 2:
            return {
                "disc_loss": 0.0,
                "disc_expert_loss": 0.0,
                "disc_policy_loss": 0.0,
                "disc_grad_pen": 0.0,
                "disc_expert_score": 0.0,
                "disc_policy_score": 0.0,
                "disc_expert_accuracy": 0.0,
                "disc_policy_accuracy": 0.0,
            }

        totals = {
            "disc_loss": 0.0,
            "disc_expert_loss": 0.0,
            "disc_policy_loss": 0.0,
            "disc_grad_pen": 0.0,
            "disc_expert_score": 0.0,
            "disc_policy_score": 0.0,
            "disc_expert_accuracy": 0.0,
            "disc_policy_accuracy": 0.0,
        }
        sample_size = min(self.amp_batch_size, self.replay_size)
        for _ in range(self.disc_updates_per_iteration):
            indices = torch.randint(
                0, self.replay_size, (sample_size,), device=self.device
            )
            policy_data = self.amp_replay_buffer[indices]
            expert_data = self._sample_expert_transitions(sample_size)
            if expert_data.shape != policy_data.shape:
                raise ValueError(
                    "Expert/policy AMP transition mismatch: {} != {}".format(
                        tuple(expert_data.shape), tuple(policy_data.shape)
                    )
                )

            total_loss, diagnostics = self.discriminator.compute_loss(
                expert_data,
                policy_data,
                self.gradient_penalty_coefficient,
            )
            self.disc_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), max_norm=10.0
            )
            self.disc_optimizer.step()
            self.discriminator_updates_completed += 1

            totals["disc_loss"] += float(total_loss.detach().item())
            totals["disc_expert_loss"] += float(
                diagnostics["expert_loss"].item()
            )
            totals["disc_policy_loss"] += float(
                diagnostics["policy_loss"].item()
            )
            totals["disc_grad_pen"] += float(
                diagnostics["gradient_penalty"].item()
            )
            totals["disc_expert_score"] += float(
                diagnostics["expert_score"].item()
            )
            totals["disc_policy_score"] += float(
                diagnostics["policy_score"].item()
            )
            totals["disc_expert_accuracy"] += float(
                diagnostics["expert_accuracy"].item()
            )
            totals["disc_policy_accuracy"] += float(
                diagnostics["policy_accuracy"].item()
            )

        divisor = float(self.disc_updates_per_iteration)
        return {key: value / divisor for key, value in totals.items()}

    def update(self):
        loss_dict = super().update()
        loss_dict.update(self._update_discriminator())
        if self._style_reward_count:
            loss_dict["amp_style_reward"] = (
                self._style_reward_sum / self._style_reward_count
            )
            loss_dict["amp_style_weight"] = (
                self._effective_style_weight_sum / self._style_reward_count
            )
        else:
            loss_dict["amp_style_reward"] = 0.0
            loss_dict["amp_style_weight"] = 0.0
        loss_dict["amp_replay_size"] = float(self.replay_size)
        self._style_reward_sum = 0.0
        self._style_reward_count = 0
        self._effective_style_weight_sum = 0.0
        return loss_dict
