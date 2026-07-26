"""On-policy runner with Adversarial Motion Prior integration."""

from __future__ import annotations

import collections
import json
import math
import os
import statistics
import time

import torch

from humanoid.algo.amp.amp_ppo import AMPPPO
from humanoid.algo.amp.discriminator import AMPDiscriminator
from humanoid.algo.amp.observations import (
    AMP_FEATURE_VERSION,
    AMP_FULL_FRAME_DIM,
    AMP_OBSERVATION_DIM,
    extract_amp_observation,
    mask_unsupported_amp_features,
)
from humanoid.algo.ppo.on_policy_runner import OnPolicyRunner
from humanoid.amp_utils.motion_loader import MotionLoaderNing
from humanoid.utils.utils import store_code_state


# Backward-compatible import used by the first AMP prototype.
extract_amp_obs = extract_amp_observation


def validate_motion_file(path, allow_legacy=False):
    """Validate the dataset before allocating a large GPU preload buffer."""
    if not os.path.isfile(path):
        raise FileNotFoundError("AMP reference motion does not exist: " + path)
    with open(path, "r") as stream:
        data = json.load(stream)
    frames = data.get("Frames")
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError(
            "AMP reference motion needs at least two frames: " + path
        )
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, list) or len(frame) != AMP_FULL_FRAME_DIM:
            raise ValueError(
                "AMP frame {} width must be {}, received {} in {}".format(
                    frame_index,
                    AMP_FULL_FRAME_DIM,
                    len(frame) if isinstance(frame, list) else "invalid",
                    path,
                )
            )
        try:
            finite = all(math.isfinite(float(value)) for value in frame)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "AMP frame {} contains a non-numeric value in {}".format(
                    frame_index, path
                )
            ) from error
        if not finite:
            raise ValueError(
                "AMP frame {} contains NaN or Inf in {}".format(
                    frame_index, path
                )
            )
    version = data.get("AMPFeatureVersion")
    if version != AMP_FEATURE_VERSION and not allow_legacy:
        raise ValueError(
            "AMP reference motion '{}' uses feature version {!r}; expected "
            "{!r}. Re-run collect_reference_motions.py, or explicitly set "
            "allow_legacy_motion_files for a verified external dataset.".format(
                path, version, AMP_FEATURE_VERSION
            )
        )
    frame_duration = float(data.get("FrameDuration", 0.0))
    if not frame_duration > 0.0:
        raise ValueError("AMP FrameDuration must be positive: " + path)
    motion_weight = float(data.get("MotionWeight", 0.0))
    if not math.isfinite(motion_weight) or motion_weight <= 0.0:
        raise ValueError("AMP MotionWeight must be positive: " + path)


class AMPOnPolicyRunner(OnPolicyRunner):
    """Standard PPO rollout collection plus an AMP discriminator."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        if self.is_distributed:
            raise NotImplementedError(
                "AMP discriminator synchronization is not implemented for "
                "multi-GPU runs; use one RL device per training process."
            )

        amp_cfg = train_cfg.get("amp")
        if not isinstance(amp_cfg, dict):
            raise ValueError("AMP runner requires a top-level 'amp' config")
        motion_files = amp_cfg.get("motion_files", [])
        if isinstance(motion_files, str):
            motion_files = [motion_files]
        motion_files = [
            os.path.abspath(os.path.expanduser(path))
            for path in motion_files
            if str(path).strip()
        ]
        if not motion_files:
            raise ValueError(
                "AMP requires at least one curated reference motion file"
            )
        allow_legacy = bool(
            amp_cfg.get("allow_legacy_motion_files", False)
        )
        for path in motion_files:
            validate_motion_file(path, allow_legacy=allow_legacy)

        preload_transitions = int(
            amp_cfg.get("num_preload_transitions", 50000)
        )
        if preload_transitions < 2:
            raise ValueError("AMP num_preload_transitions must be at least 2")
        print("[AMP] Loading {} reference motion(s)".format(len(motion_files)))
        self.motion_loader = MotionLoaderNing(
            device=self.device,
            time_between_frames=float(getattr(env, "dt", 0.02)),
            reference_observation_horizon=2,
            num_preload_transitions=preload_transitions,
            motion_files=motion_files,
        )
        if self.motion_loader.observation_dim != AMP_OBSERVATION_DIM:
            raise ValueError(
                "MotionLoader AMP width is {}, expected {}".format(
                    self.motion_loader.observation_dim,
                    AMP_OBSERVATION_DIM,
                )
            )

        expert_observations = self.motion_loader.preloaded_states[
            :, :, self.motion_loader.observation_start_dim :
        ]
        expert_observations = mask_unsupported_amp_features(
            expert_observations
        )
        expert_mean = expert_observations.mean(dim=(0, 1))
        expert_std = expert_observations.std(
            dim=(0, 1), unbiased=False
        )
        del expert_observations

        discriminator = AMPDiscriminator(
            input_dim=AMP_OBSERVATION_DIM * 2,
            hidden_dims=amp_cfg.get("disc_hidden_dims", (1024, 512)),
        ).to(self.device)
        old_alg = self.alg
        self.alg = AMPPPO(
            policy=old_alg.policy,
            discriminator=discriminator,
            expert_observation_mean=expert_mean,
            expert_observation_std=expert_std,
            num_learning_epochs=old_alg.num_learning_epochs,
            num_mini_batches=old_alg.num_mini_batches,
            clip_param=old_alg.clip_param,
            gamma=old_alg.gamma,
            lam=old_alg.lam,
            value_loss_coef=old_alg.value_loss_coef,
            entropy_coef=old_alg.entropy_coef,
            learning_rate=old_alg.learning_rate,
            max_grad_norm=old_alg.max_grad_norm,
            use_clipped_value_loss=old_alg.use_clipped_value_loss,
            schedule=old_alg.schedule,
            desired_kl=old_alg.desired_kl,
            device=self.device,
            normalize_advantage_per_mini_batch=(
                old_alg.normalize_advantage_per_mini_batch
            ),
            symmetry_cfg=old_alg.symmetry,
            multi_gpu_cfg=self.multi_gpu_cfg,
            amp_replay_buffer_size=amp_cfg.get(
                "replay_buffer_size", 100000
            ),
            amp_batch_size=amp_cfg.get("batch_size", 512),
            style_reward_weight=amp_cfg.get(
                "style_reward_weight", 0.2
            ),
            disc_learning_rate=amp_cfg.get(
                "disc_learning_rate", 1.0e-4
            ),
            disc_updates_per_iteration=amp_cfg.get(
                "disc_updates_per_iteration", 2
            ),
            gradient_penalty_coefficient=amp_cfg.get(
                "gradient_penalty_coefficient", 10.0
            ),
            reward_warmup_updates=amp_cfg.get(
                "reward_warmup_updates", 10
            ),
            reward_ramp_updates=amp_cfg.get(
                "reward_ramp_updates", 100
            ),
        )
        self.alg.motion_loader = self.motion_loader
        self.discriminator = self.alg.discriminator
        self.alg.set_surrogate_loss_scale(old_alg.surrogate_loss_scale)
        if old_alg.actor_reference is not None:
            self.alg.actor_reference = old_alg.actor_reference
            self.alg.actor_reference_loss_coeff = (
                old_alg.actor_reference_loss_coeff
            )
            self.alg.actor_reference_symmetry_env = (
                old_alg.actor_reference_symmetry_env
            )
            self.alg.actor_reference_mirror_blend = (
                old_alg.actor_reference_mirror_blend
            )

        observations = self.env.get_observations()
        privileged_observations = self.env.get_privileged_observations()
        critic_width = (
            privileged_observations.shape[1]
            if privileged_observations is not None
            else observations.shape[1]
        )
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [observations.shape[1]],
            [critic_width],
            [self.env.num_actions],
        )
        self.best_mean_task_reward = float("-inf")
        print(
            "[AMP] Ready: obs_dim={} motions={} preload={} style_weight={:.3f} "
            "warmup={} ramp={}".format(
                AMP_OBSERVATION_DIM,
                self.motion_loader.num_motions,
                preload_transitions,
                self.alg.style_reward_weight,
                self.alg.reward_warmup_updates,
                self.alg.reward_ramp_updates,
            )
        )

    def _initialize_writer(self):
        if self.log_dir is None or self.writer is not None or self.disable_logs:
            return
        self.logger_type = self.cfg.get("logger", "tensorboard").lower()
        if self.logger_type != "tensorboard":
            raise ValueError(
                "AMP runner currently supports the tensorboard logger"
            )
        from torch.utils.tensorboard import SummaryWriter

        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        config_path = os.path.join(self.log_dir, "train_cfg.json")
        symmetry_cfg = self.alg_cfg.get("symmetry_cfg")
        symmetry_env = None
        if isinstance(symmetry_cfg, dict):
            symmetry_env = symmetry_cfg.pop("_env", None)
        try:
            with open(config_path, "w") as stream:
                json.dump(self.cfg, stream, indent=4)
        finally:
            if symmetry_env is not None:
                symmetry_cfg["_env"] = symmetry_env

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        self._initialize_writer()
        if self.log_dir is not None and not self.disable_logs:
            # A run always has a recoverable baseline.  Later completed-task
            # reward improvements atomically replace this file.
            self.save(
                os.path.join(self.log_dir, "model_best.pt"),
                infos={
                    "selection_metric": "initial_policy_baseline",
                    "selection_value": None,
                },
            )
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        observations = self.env.get_observations().to(self.device)
        privileged_observations = self.env.get_privileged_observations()
        if privileged_observations is None:
            privileged_observations = observations
        else:
            privileged_observations = privileged_observations.to(self.device)
        observations = self.obs_normalizer(observations)
        self.train_mode()

        ep_infos = []
        reward_buffer = collections.deque(maxlen=100)
        length_buffer = collections.deque(maxlen=100)
        current_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        current_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        start_iter = self.current_learning_iteration
        total_iter = start_iter + int(num_learning_iterations)
        for iteration in range(start_iter, total_iter):
            collection_start = time.time()
            iteration_completed_rewards = []
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    amp_observation = extract_amp_observation(self.env).to(
                        self.device
                    )
                    actions = self.alg.act(
                        observations, privileged_observations
                    )
                    (
                        next_observations,
                        next_privileged_observations,
                        rewards,
                        dones,
                        infos,
                        _,
                        _,
                    ) = self.env.step(actions.to(self.env.device))
                    amp_next_observation = extract_amp_observation(self.env).to(
                        self.device
                    )
                    next_observations = next_observations.to(self.device)
                    if next_privileged_observations is None:
                        next_privileged_observations = next_observations
                    else:
                        next_privileged_observations = (
                            next_privileged_observations.to(self.device)
                        )
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    self.alg.transition.amp_obs = amp_observation
                    self.alg.transition.amp_obs_next = amp_next_observation
                    observations = self.obs_normalizer(next_observations)
                    privileged_observations = next_privileged_observations
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        current_reward_sum += rewards
                        current_episode_length += 1
                        done_ids = torch.nonzero(
                            dones.reshape(-1) > 0, as_tuple=False
                        ).flatten()
                        if done_ids.numel():
                            completed_rewards = (
                                current_reward_sum[done_ids]
                                .detach()
                                .cpu()
                                .tolist()
                            )
                            completed_lengths = (
                                current_episode_length[done_ids]
                                .detach()
                                .cpu()
                                .tolist()
                            )
                            reward_buffer.extend(completed_rewards)
                            length_buffer.extend(completed_lengths)
                            iteration_completed_rewards.extend(
                                completed_rewards
                            )
                            current_reward_sum[done_ids] = 0.0
                            current_episode_length[done_ids] = 0.0

                collection_time = time.time() - collection_start
                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_observations)

            learning_start = time.time()
            loss_dict = self.alg.update()
            learn_time = time.time() - learning_start
            self.current_learning_iteration = iteration + 1

            # Names expected by OnPolicyRunner.log().
            it = iteration
            tot_iter = total_iter
            num_learning_iterations = int(num_learning_iterations)
            rewbuffer = reward_buffer
            lenbuffer = length_buffer
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if (
                    self.current_learning_iteration % self.save_interval == 0
                ):
                    self.save(
                        os.path.join(
                            self.log_dir,
                            "model_{}.pt".format(
                                self.current_learning_iteration
                            ),
                        )
                    )
                if iteration_completed_rewards:
                    mean_task_reward = statistics.mean(
                        iteration_completed_rewards
                    )
                    if mean_task_reward > self.best_mean_task_reward:
                        self.best_mean_task_reward = mean_task_reward
                        self.save(
                            os.path.join(self.log_dir, "model_best.pt"),
                            infos={
                                "selection_metric": "mean_task_reward",
                                "selection_value": mean_task_reward,
                            },
                        )
                        print(
                            "[AMP] New task-reward best: {:.4f} at iteration "
                            "{}".format(
                                mean_task_reward,
                                self.current_learning_iteration,
                            )
                        )
            ep_infos.clear()

            if (
                iteration == start_iter
                and self.log_dir is not None
                and not self.disable_logs
            ):
                git_paths = store_code_state(
                    self.log_dir, self.git_status_repos
                )
                if (
                    getattr(self, "logger_type", "tensorboard")
                    in ("wandb", "neptune")
                    and git_paths
                ):
                    for path in git_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(
                os.path.join(
                    self.log_dir,
                    "model_{}.pt".format(self.current_learning_iteration),
                )
            )

    def save(self, path, infos=None):
        saved = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
            "amp_feature_version": AMP_FEATURE_VERSION,
            "amp_discriminator_state_dict": (
                self.alg.discriminator.state_dict()
            ),
            "amp_optimizer_state_dict": self.alg.disc_optimizer.state_dict(),
            "amp_discriminator_updates": (
                self.alg.discriminator_updates_completed
            ),
            "amp_best_mean_task_reward": self.best_mean_task_reward,
            "amp_expert_observation_mean": (
                self.alg.expert_observation_mean.detach().cpu()
            ),
            "amp_expert_observation_std": (
                self.alg.expert_observation_std.detach().cpu()
            ),
        }
        if hasattr(self.env, "get_checkpoint_state"):
            saved["env_state"] = self.env.get_checkpoint_state()
        if self.empirical_normalization:
            saved["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved["privileged_obs_norm_state_dict"] = (
                self.privileged_obs_normalizer.state_dict()
            )
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        temporary_path = "{}.tmp.{}".format(path, os.getpid())
        torch.save(saved, temporary_path)
        os.replace(temporary_path, path)
        if (
            getattr(self, "logger_type", "tensorboard")
            in ("neptune", "wandb")
            and not self.disable_logs
            and self.writer is not None
        ):
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        infos = super().load(path, load_optimizer=load_optimizer)
        try:
            loaded = torch.load(
                path, weights_only=False, map_location=self.device
            )
        except TypeError:
            loaded = torch.load(path, map_location=self.device)
        discriminator_state = loaded.get("amp_discriminator_state_dict")
        if discriminator_state is None:
            print(
                "[AMP] Base PPO checkpoint loaded; discriminator starts fresh."
            )
        else:
            version = loaded.get("amp_feature_version")
            if version != AMP_FEATURE_VERSION:
                raise ValueError(
                    "AMP checkpoint feature version {!r} is incompatible "
                    "with {!r}".format(version, AMP_FEATURE_VERSION)
                )
            self.alg.discriminator.load_state_dict(discriminator_state)
            saved_mean = loaded.get("amp_expert_observation_mean")
            saved_std = loaded.get("amp_expert_observation_std")
            if saved_mean is not None and saved_std is not None:
                saved_mean = torch.as_tensor(
                    saved_mean,
                    dtype=torch.float32,
                    device=self.device,
                ).reshape(-1)
                saved_std = torch.as_tensor(
                    saved_std,
                    dtype=torch.float32,
                    device=self.device,
                ).reshape(-1)
                if (
                    saved_mean.numel() != AMP_OBSERVATION_DIM
                    or saved_std.numel() != AMP_OBSERVATION_DIM
                    or not torch.isfinite(saved_mean).all()
                    or not torch.isfinite(saved_std).all()
                    or torch.any(saved_std <= 0.0)
                ):
                    raise ValueError(
                        "AMP checkpoint has invalid expert normalization state"
                    )
                self.alg.expert_observation_mean.copy_(saved_mean)
                self.alg.expert_observation_std.copy_(saved_std)
            if load_optimizer and "amp_optimizer_state_dict" in loaded:
                self.alg.disc_optimizer.load_state_dict(
                    loaded["amp_optimizer_state_dict"]
                )
            self.alg.discriminator_updates_completed = int(
                loaded.get("amp_discriminator_updates", 0)
            )
            self.best_mean_task_reward = float(
                loaded.get("amp_best_mean_task_reward", float("-inf"))
            )
            print(
                "[AMP] Restored discriminator at update {}".format(
                    self.alg.discriminator_updates_completed
                )
            )
        return infos

    def train_mode(self):
        super().train_mode()
        self.alg.discriminator.train()

    def eval_mode(self):
        super().eval_mode()
        self.alg.discriminator.eval()
