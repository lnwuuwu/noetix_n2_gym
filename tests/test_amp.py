"""CPU-only tests for the AMP data contract and auxiliary algorithm."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from humanoid.algo.amp.amp_ppo import AMPPPO
from humanoid.algo.amp.amp_runner import (
    AMPOnPolicyRunner,
    validate_motion_file,
)
from humanoid.algo.amp.amp_storage import AMPRolloutStorage
from humanoid.algo.amp.discriminator import AMPDiscriminator
from humanoid.algo.amp.observations import (
    AMP_FEATURE_VERSION,
    AMP_FULL_FRAME_DIM,
    AMP_MASKED_OBSERVATION_INDICES,
    AMP_OBSERVATION_DIM,
    build_amp_full_frame,
    extract_amp_observation,
    mask_unsupported_amp_features,
    mirror_amp_full_frames,
)
from humanoid.amp_utils.motion_loader import (
    MotionLoaderNing,
    quaternion_slerp,
)


ROOT = Path(__file__).resolve().parents[1]


class _FakeAMPEnvironment:
    def __init__(self):
        self.num_envs = 2
        self.device = "cpu"
        self.body_names = [
            "L_foot",
            "R_foot",
            "L_hand",
            "R_hand",
        ]
        self.feet_indices = torch.tensor([0, 1])
        self.root_states = torch.zeros(2, 13)
        self.root_states[:, :3] = torch.tensor(
            [[1.0, 2.0, 0.7], [-1.0, 3.0, 0.8]]
        )
        self.root_states[:, 6] = 1.0
        local = torch.tensor(
            [
                [0.2, 0.1, -0.6],
                [0.2, -0.1, -0.6],
                [0.0, 0.3, 0.2],
                [0.0, -0.3, 0.2],
            ]
        )
        self.rigid_body_states_view = torch.zeros(2, 4, 13)
        self.rigid_body_states_view[:, :, :3] = (
            self.root_states[:, None, :3] + local[None, :, :]
        )
        self.dof_pos = torch.arange(36, dtype=torch.float32).reshape(2, 18)
        self.dof_vel = self.dof_pos * 0.01
        self.base_lin_vel = torch.tensor(
            [[0.2, 0.0, 0.0], [0.3, 0.1, 0.0]]
        )
        self.base_ang_vel = torch.tensor(
            [[0.0, 0.1, 0.0], [0.1, 0.0, 0.2]]
        )
        self.terrain_h = torch.tensor([0.1, 0.2])


class _DummyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(3, 2)
        self.critic = torch.nn.Linear(3, 1)


class _FakeTrainingEnvironment(_FakeAMPEnvironment):
    def __init__(self):
        super().__init__()
        self.num_envs = 4
        self.num_actions = 18
        self.device = "cpu"
        self.dt = 0.02
        self.max_episode_length = 8
        self.episode_length_buf = torch.zeros(4, dtype=torch.long)
        self._steps = torch.zeros(4, dtype=torch.long)
        self._observations = torch.zeros(4, 6)
        self._privileged_observations = torch.zeros(4, 5)
        self.root_states = torch.zeros(4, 13)
        self.root_states[:, 2] = 0.7
        self.root_states[:, 6] = 1.0
        local = torch.tensor(
            [
                [0.2, 0.1, -0.6],
                [0.2, -0.1, -0.6],
                [0.0, 0.3, 0.2],
                [0.0, -0.3, 0.2],
            ]
        )
        self.rigid_body_states_view = torch.zeros(4, 4, 13)
        self.rigid_body_states_view[:, :, :3] = (
            self.root_states[:, None, :3] + local[None]
        )
        self.dof_pos = torch.zeros(4, 18)
        self.dof_vel = torch.zeros(4, 18)
        self.base_lin_vel = torch.zeros(4, 3)
        self.base_ang_vel = torch.zeros(4, 3)
        self.terrain_h = torch.zeros(4)

    def get_observations(self):
        return self._observations

    def get_privileged_observations(self):
        return self._privileged_observations

    def reset(self):
        self._steps.zero_()
        self.episode_length_buf.zero_()
        self._observations.zero_()
        self._privileged_observations.zero_()
        return self._observations, self._privileged_observations

    def step(self, actions):
        self._steps += 1
        self.episode_length_buf += 1
        action_mean = actions.mean(dim=1)
        self._observations = torch.stack(
            (
                action_mean,
                torch.sin(self._steps.float()),
                torch.cos(self._steps.float()),
                self._steps.float() * 0.01,
                actions[:, 0],
                actions[:, 1],
            ),
            dim=1,
        )
        self._privileged_observations = self._observations[:, :5].clone()
        self.dof_vel = actions.detach().clone()
        self.dof_pos = self.dof_pos + self.dt * self.dof_vel
        self.base_lin_vel[:, 0] = action_mean.detach()
        self.root_states[:, 0] += self.dt * self.base_lin_vel[:, 0]
        self.rigid_body_states_view[:, :, :3] += (
            self.dt * self.base_lin_vel[:, None, :]
        )
        rewards = 1.0 - 0.01 * torch.mean(actions.square(), dim=1)
        dones = self._steps.remainder(3) == 0
        termination_ids = torch.nonzero(
            dones, as_tuple=False
        ).flatten()
        self._steps[dones] = 0
        self.episode_length_buf[dones] = 0
        return (
            self._observations,
            self._privileged_observations,
            rewards,
            dones,
            {},
            termination_ids,
            None,
        )

    def get_checkpoint_state(self):
        return {"steps": self._steps.clone()}

    def load_checkpoint_state(self, state):
        self._steps.copy_(state["steps"])


def _make_amp_ppo(replay_size=5, warmup=2, ramp=4):
    return AMPPPO(
        policy=_DummyPolicy(),
        discriminator=AMPDiscriminator(110, (16, 8)),
        expert_observation_mean=torch.zeros(AMP_OBSERVATION_DIM),
        expert_observation_std=torch.ones(AMP_OBSERVATION_DIM),
        amp_replay_buffer_size=replay_size,
        amp_batch_size=2,
        reward_warmup_updates=warmup,
        reward_ramp_updates=ramp,
        style_reward_weight=0.2,
    )


class AMPObservationTests(unittest.TestCase):
    def test_full_frame_contract_and_translation_invariance(self):
        env = _FakeAMPEnvironment()
        full = build_amp_full_frame(env)
        observation = extract_amp_observation(env)
        self.assertEqual(full.shape, (2, AMP_FULL_FRAME_DIM))
        self.assertEqual(observation.shape, (2, AMP_OBSERVATION_DIM))
        expected_local = torch.tensor(
            [
                [0.2, 0.1, -0.6],
                [0.2, -0.1, -0.6],
                [0.0, 0.3, 0.2],
                [0.0, -0.3, 0.2],
            ]
        ).reshape(-1)
        torch.testing.assert_close(full[0, 25:37], expected_local)
        torch.testing.assert_close(
            full[:, 61], torch.tensor([0.6, 0.6])
        )

        before = observation.clone()
        translation = torch.tensor([4.0, -7.0, 0.0])
        env.root_states[:, :3] += translation
        env.rigid_body_states_view[:, :, :3] += translation
        torch.testing.assert_close(extract_amp_observation(env), before)

    def test_mask_is_identical_and_non_mutating(self):
        observation = torch.ones(3, AMP_OBSERVATION_DIM)
        masked = mask_unsupported_amp_features(observation)
        self.assertTrue(torch.all(observation == 1.0))
        self.assertTrue(
            torch.all(
                masked[:, list(AMP_MASKED_OBSERVATION_INDICES)] == 0.0
            )
        )

    def test_mirror_is_an_involution(self):
        frames = torch.randn(4, AMP_FULL_FRAME_DIM)
        frames[:, 3:7] = torch.nn.functional.normalize(
            frames[:, 3:7], dim=1
        )

        def mirror_joints(values):
            return -torch.flip(values, dims=(-1,))

        mirrored = mirror_amp_full_frames(frames, mirror_joints)
        restored = mirror_amp_full_frames(mirrored, mirror_joints)
        torch.testing.assert_close(restored, frames)


class AMPStorageAndAlgorithmTests(unittest.TestCase):
    def test_storage_generator_preserves_ppo_eleven_item_interface(self):
        storage = AMPRolloutStorage(
            "rl", 2, 4, [3], [4], [2], device="cpu"
        )
        storage.observations.normal_()
        storage.privileged_observations.normal_()
        storage.actions.normal_()
        storage.values.normal_()
        storage.returns.normal_()
        storage.advantages.normal_()
        storage.actions_log_prob.normal_()
        storage.mu.normal_()
        storage.sigma.fill_(0.5)
        batch = next(storage.mini_batch_generator(2, 1))
        self.assertEqual(len(batch), 11)
        self.assertEqual(batch[9], (None, None))
        self.assertIsNone(batch[10])

    def test_replay_buffer_masks_done_and_handles_oversized_input(self):
        algorithm = _make_amp_ppo(replay_size=5)
        first = torch.arange(7 * 110, dtype=torch.float32).reshape(7, 110)
        algorithm._update_replay_buffer(first)
        self.assertEqual(algorithm.replay_size, 5)
        torch.testing.assert_close(algorithm.amp_replay_buffer, first[-5:])

        second = torch.full((3, 110), -1.0)
        algorithm._update_replay_buffer(
            second, valid_mask=torch.tensor([True, False, True])
        )
        self.assertEqual(algorithm.replay_size, 5)
        self.assertTrue(algorithm.amp_replay_full)

    def test_style_reward_warmup_and_linear_ramp(self):
        algorithm = _make_amp_ppo(warmup=2, ramp=4)
        expected = {
            0: 0.0,
            1: 0.0,
            2: 0.05,
            3: 0.10,
            4: 0.15,
            5: 0.20,
            9: 0.20,
        }
        for updates, weight in expected.items():
            algorithm.discriminator_updates_completed = updates
            self.assertAlmostEqual(
                algorithm.effective_style_reward_weight, weight
            )

    def test_policy_and_expert_mask_use_the_same_feature_positions(self):
        algorithm = _make_amp_ppo()
        state = torch.ones(2, AMP_OBSERVATION_DIM)
        transition = algorithm._normalized_transition(state, state)
        for index in AMP_MASKED_OBSERVATION_INDICES:
            self.assertTrue(torch.all(transition[:, index] == 0.0))
            self.assertTrue(
                torch.all(
                    transition[:, AMP_OBSERVATION_DIM + index] == 0.0
                )
            )

    def test_discriminator_loss_is_finite_and_backpropagates(self):
        discriminator = AMPDiscriminator(110, (32, 16))
        expert = torch.randn(8, 110)
        policy = torch.randn(8, 110) + 0.5
        loss, diagnostics = discriminator.compute_loss(
            expert, policy, gradient_penalty_coefficient=1.0
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                for parameter in discriminator.parameters()
            )
        )
        self.assertEqual(
            set(diagnostics),
            {
                "expert_loss",
                "policy_loss",
                "gradient_penalty",
                "expert_score",
                "policy_score",
                "expert_accuracy",
                "policy_accuracy",
            },
        )


class AMPFullRunnerIntegrationTests(unittest.TestCase):
    @staticmethod
    def _motion_file(directory):
        path = Path(directory) / "reference.json"
        frames = np.zeros((8, AMP_FULL_FRAME_DIM), dtype=np.float32)
        frames[:, 6] = 1.0
        frames[:, 7:] = np.linspace(0.0, 0.2, 8)[:, None]
        path.write_text(
            json.dumps(
                {
                    "Frames": frames.tolist(),
                    "MotionWeight": 1.0,
                    "FrameDuration": 0.02,
                    "AMPFeatureVersion": AMP_FEATURE_VERSION,
                }
            )
        )
        return str(path)

    @staticmethod
    def _config(motion_file):
        return {
            "policy": {
                "class_name": "ActorCritic",
                "actor_hidden_dims": [16, 8],
                "critic_hidden_dims": [16, 8],
                "activation": "elu",
                "init_noise_std": 0.2,
                "noise_std_type": "scalar",
            },
            "algorithm": {
                "class_name": "PPO",
                "value_loss_coef": 1.0,
                "use_clipped_value_loss": True,
                "clip_param": 0.2,
                "entropy_coef": 0.0,
                "num_learning_epochs": 1,
                "num_mini_batches": 2,
                "learning_rate": 1.0e-3,
                "schedule": "fixed",
                "gamma": 0.99,
                "lam": 0.95,
                "desired_kl": 0.01,
                "max_grad_norm": 1.0,
            },
            "runner": {
                "num_steps_per_env": 4,
                "save_interval": 1,
                "empirical_normalization": False,
            },
            "amp": {
                "motion_files": [motion_file],
                "num_preload_transitions": 16,
                "disc_hidden_dims": [16, 8],
                "replay_buffer_size": 32,
                "batch_size": 4,
                "disc_updates_per_iteration": 1,
                "gradient_penalty_coefficient": 0.1,
                "style_reward_weight": 0.1,
                "reward_warmup_updates": 0,
                "reward_ramp_updates": 1,
            },
        }

    def test_two_iteration_runner_update_save_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            motion_file = self._motion_file(temporary)
            runner = AMPOnPolicyRunner(
                _FakeTrainingEnvironment(),
                self._config(motion_file),
                log_dir=None,
                device="cpu",
            )
            runner.learn(2)
            self.assertEqual(runner.current_learning_iteration, 2)
            self.assertEqual(
                runner.alg.discriminator_updates_completed, 2
            )
            self.assertGreater(runner.alg.replay_size, 0)

            checkpoint = str(Path(temporary) / "model_2.pt")
            runner.save(checkpoint)
            saved = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            self.assertIn("amp_discriminator_state_dict", saved)
            self.assertIn("amp_optimizer_state_dict", saved)
            self.assertIn("amp_expert_observation_mean", saved)

            resumed = AMPOnPolicyRunner(
                _FakeTrainingEnvironment(),
                self._config(motion_file),
                log_dir=None,
                device="cpu",
            )
            resumed.load(checkpoint)
            self.assertEqual(resumed.current_learning_iteration, 2)
            self.assertEqual(
                resumed.alg.discriminator_updates_completed, 2
            )


class MotionLoaderTests(unittest.TestCase):
    def test_motion_file_validation_checks_every_frame_and_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "motion.json"
            valid_frame = [0.0] * AMP_FULL_FRAME_DIM
            valid_frame[6] = 1.0
            payload = {
                "Frames": [valid_frame, list(valid_frame)],
                "MotionWeight": 1.0,
                "FrameDuration": 0.02,
                "AMPFeatureVersion": AMP_FEATURE_VERSION,
            }
            path.write_text(json.dumps(payload))
            validate_motion_file(str(path))

            payload["Frames"][1] = payload["Frames"][1][:-1]
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "frame 1 width"):
                validate_motion_file(str(path))

            payload["Frames"] = [valid_frame, list(valid_frame)]
            payload["AMPFeatureVersion"] = "legacy"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "feature version"):
                validate_motion_file(str(path))

    def test_quaternion_slerp_has_exact_endpoints_and_unit_midpoint(self):
        start = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        end = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
        at_start = quaternion_slerp(start, end, torch.tensor([0.0]))
        at_half = quaternion_slerp(start, end, torch.tensor([0.5]))
        at_end = quaternion_slerp(start, end, torch.tensor([1.0]))
        torch.testing.assert_close(at_start, start, atol=2e-6, rtol=0)
        torch.testing.assert_close(at_end, end, atol=2e-6, rtol=0)
        torch.testing.assert_close(
            torch.linalg.vector_norm(at_half, dim=1),
            torch.ones(1),
            atol=2e-6,
            rtol=0,
        )

    def test_motion_loader_interpolates_by_n_minus_one_intervals(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "motion.json"
            frames = np.zeros((3, AMP_FULL_FRAME_DIM), dtype=np.float32)
            frames[:, 6] = 1.0
            frames[0, 7:] = 0.0
            frames[1, 7:] = 1.0
            frames[2, 7:] = 2.0
            path.write_text(
                json.dumps(
                    {
                        "Frames": frames.tolist(),
                        "MotionWeight": 1.0,
                        "FrameDuration": 0.1,
                        "AMPFeatureVersion": AMP_FEATURE_VERSION,
                    }
                )
            )
            loader = MotionLoaderNing(
                device="cpu",
                time_between_frames=0.1,
                reference_observation_horizon=2,
                num_preload_transitions=4,
                motion_files=[str(path)],
            )
            middle = loader.get_full_frame_at_time(0, 0.1)
            torch.testing.assert_close(
                middle[7:], torch.ones(AMP_OBSERVATION_DIM)
            )


class AMPStaticIntegrationTests(unittest.TestCase):
    def test_launcher_invokes_amp_entry_and_keeps_task_rewards(self):
        launcher = (
            ROOT / "humanoid/scripts/run_amp_training.sh"
        ).read_text()
        self.assertIn("humanoid/scripts/train_amp.py", launcher)
        self.assertIn("--motion_manifest", launcher)
        self.assertIn("--reward_scale_overrides", launcher)
        self.assertIn("stairs_right_stride_excess=-10", launcher)
        self.assertIn("stairs_right_support_stability=-8", launcher)
        self.assertIn(
            'ACTOR_LAYERS="${N2_AMP_ACTOR_LAYERS:-4}"', launcher
        )
        self.assertIn(
            'OBSERVATION_NOISE="${N2_AMP_OBSERVATION_NOISE:-0.03}"',
            launcher,
        )
        self.assertNotIn("STYLE_ZERO_OVERRIDES", launcher)
        self.assertNotIn("humanoid/scripts/train.py \\", launcher)

    def test_training_entry_uses_one_standard_train_path(self):
        entry = (ROOT / "humanoid/scripts/train_amp.py").read_text()
        self.assertIn('runner_class_name = "AMPOnPolicyRunner"', entry)
        self.assertIn("train(args)", entry)
        self.assertNotIn("ppo_runner", entry)
        self.assertNotIn("AMPOnPolicyRunner(", entry)

    def test_collector_writes_versioned_episode_files(self):
        collector = (
            ROOT / "humanoid/scripts/collect_reference_motions.py"
        ).read_text()
        self.assertIn('"AMPFeatureVersion": AMP_FEATURE_VERSION', collector)
        self.assertIn(
            'open(manifest, "w", encoding="utf-8")',
            collector,
        )
        self.assertIn(
            'open(path, "w", encoding="utf-8")',
            collector,
        )
        self.assertIn("last_episode_completion", collector)
        self.assertIn("--max_final_lateral_position", collector)
        self.assertIn('"default": 0.06', collector)
        self.assertIn("episode_frames[env_id] = []", collector)
        self.assertNotIn("all_frames.extend", collector)

    def test_amp_text_io_is_explicitly_utf8(self):
        runner = (
            ROOT / "humanoid/algo/amp/amp_runner.py"
        ).read_text()
        loader = (
            ROOT / "humanoid/amp_utils/motion_loader.py"
        ).read_text()
        entry = (ROOT / "humanoid/scripts/train_amp.py").read_text()
        self.assertIn('open(path, "r", encoding="utf-8")', runner)
        self.assertEqual(
            loader.count('open(motion_file, "r", encoding="utf-8")'),
            4,
        )
        self.assertIn(
            'open(manifest, "r", encoding="utf-8")',
            entry,
        )


if __name__ == "__main__":
    unittest.main()
