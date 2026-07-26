import os
import tempfile
import unittest

import torch

from humanoid.utils.policy_bootstrap import (
    FASTSTAIR_FRAME_SIZE,
    FRAME_STACK,
    LEGACY_FRAME_SIZE,
    SHARED_PROPRIO_SIZE,
    bootstrap_faststair_actor,
    nearest_terrain_mapping,
)


def actor(input_width):
    return torch.nn.Sequential(
        torch.nn.Linear(input_width, 16),
        torch.nn.ELU(),
        torch.nn.Linear(16, 8),
        torch.nn.ELU(),
        torch.nn.Linear(8, 4),
    )


class DummyPolicy(torch.nn.Module):
    def __init__(self, actor_width):
        super().__init__()
        self.actor = actor(actor_width)
        self.critic = torch.nn.Linear(7, 1)
        self.std = torch.nn.Parameter(torch.ones(4))


class FastStairActorBootstrapTests(unittest.TestCase):
    def _checkpoint(self, directory, source):
        path = os.path.join(directory, "model_9440.pt")
        torch.save(
            {
                "model_state_dict": source.state_dict(),
                "iter": 9440,
            },
            path,
        )
        return path

    def test_transfer_preserves_actor_on_mapped_observations(self):
        torch.manual_seed(7)
        source = DummyPolicy(LEGACY_FRAME_SIZE * FRAME_STACK)
        target = DummyPolicy(FASTSTAIR_FRAME_SIZE * FRAME_STACK)
        critic_before = {
            key: value.clone()
            for key, value in target.critic.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            report = bootstrap_faststair_actor(
                target, self._checkpoint(directory, source)
            )

        legacy_obs = torch.randn(13, LEGACY_FRAME_SIZE * FRAME_STACK)
        faststair_obs = torch.zeros(
            13, FASTSTAIR_FRAME_SIZE * FRAME_STACK
        )
        mapping = nearest_terrain_mapping()
        for frame in range(FRAME_STACK):
            source_offset = frame * LEGACY_FRAME_SIZE
            target_offset = frame * FASTSTAIR_FRAME_SIZE
            faststair_obs[
                :,
                target_offset : target_offset + SHARED_PROPRIO_SIZE,
            ] = legacy_obs[
                :,
                source_offset : source_offset + SHARED_PROPRIO_SIZE,
            ]
            for source_index, target_index in enumerate(mapping):
                faststair_obs[
                    :,
                    target_offset + SHARED_PROPRIO_SIZE + target_index,
                ] = legacy_obs[
                    :,
                    source_offset + SHARED_PROPRIO_SIZE + source_index,
                ]

        self.assertTrue(
            torch.allclose(
                source.actor(legacy_obs),
                target.actor(faststair_obs),
                atol=1.0e-6,
                rtol=1.0e-6,
            )
        )
        self.assertTrue(torch.equal(source.std, target.std))
        for key, value in target.critic.state_dict().items():
            self.assertTrue(torch.equal(value, critic_before[key]))
        self.assertEqual(report["source_iteration"], 9440)
        self.assertEqual(report["mapped_terrain_per_frame"], 12)
        self.assertEqual(report["new_terrain_per_frame"], 33)

    def test_unmapped_faststair_inputs_start_at_zero(self):
        source = DummyPolicy(LEGACY_FRAME_SIZE * FRAME_STACK)
        target = DummyPolicy(FASTSTAIR_FRAME_SIZE * FRAME_STACK)
        with tempfile.TemporaryDirectory() as directory:
            bootstrap_faststair_actor(
                target, self._checkpoint(directory, source)
            )
        mapped = set(nearest_terrain_mapping())
        first_weight = target.actor[0].weight
        for frame in range(FRAME_STACK):
            offset = frame * FASTSTAIR_FRAME_SIZE + SHARED_PROPRIO_SIZE
            for terrain_index in range(45):
                if terrain_index not in mapped:
                    self.assertEqual(
                        torch.count_nonzero(
                            first_weight[:, offset + terrain_index]
                        ).item(),
                        0,
                    )

    def test_rejects_nonlegacy_source_width(self):
        source = DummyPolicy(400)
        target = DummyPolicy(FASTSTAIR_FRAME_SIZE * FRAME_STACK)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self._checkpoint(directory, source)
            with self.assertRaisesRegex(ValueError, "Legacy Actor input"):
                bootstrap_faststair_actor(target, checkpoint)


if __name__ == "__main__":
    unittest.main()
