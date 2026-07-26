import unittest

import torch

from humanoid.utils.faststair_planner import (
    DCMPlannerWeights,
    dcm_foothold_search,
    rectangular_search_offsets,
)


class FastStairPlannerTests(unittest.TestCase):
    def _search(self, **overrides):
        candidates = torch.tensor(
            [
                [
                    [0.26, -0.06, 0.10],
                    [0.30, 0.00, 0.10],
                    [0.34, 0.06, 0.10],
                ]
            ],
            dtype=torch.float32,
        )
        values = {
            "com_xy": torch.tensor([[0.0, 0.0]]),
            "com_velocity_xy": torch.tensor([[0.0, 0.0]]),
            "com_height": torch.tensor([0.70]),
            "stance_xy": torch.tensor([[0.0, 0.0]]),
            "desired_velocity_xy": torch.tensor([[0.0, 0.0]]),
            "nominal_foothold": torch.tensor([[0.30, 0.00, 0.10]]),
            "candidates": candidates,
            "candidate_valid": torch.ones(1, 3, dtype=torch.bool),
            "horizon": torch.tensor([0.30]),
            "weights": DCMPlannerWeights(
                nominal=1.0,
                dcm_offset=0.0,
                steepness=0.0,
                edge=0.0,
            ),
        }
        values.update(overrides)
        return dcm_foothold_search(**values)

    def test_rectangular_offsets_are_deterministic(self):
        offsets = rectangular_search_offsets(
            [-0.02, 0.02], [-0.03, 0.0, 0.03]
        )
        self.assertEqual(tuple(offsets.shape), (6, 2))
        self.assertTrue(
            torch.equal(
                offsets,
                torch.tensor(
                    [
                        [-0.02, -0.03],
                        [-0.02, 0.00],
                        [-0.02, 0.03],
                        [0.02, -0.03],
                        [0.02, 0.00],
                        [0.02, 0.03],
                    ]
                ),
            )
        )

    def test_nominal_cost_selects_tread_center(self):
        result = self._search()
        self.assertTrue(result.valid.item())
        self.assertEqual(result.selected_index.item(), 1)
        self.assertTrue(
            torch.allclose(
                result.foothold,
                torch.tensor([[0.30, 0.00, 0.10]]),
            )
        )

    def test_no_valid_candidate_uses_exact_nominal_fallback(self):
        nominal = torch.tensor([[0.31, -0.04, 0.12]])
        result = self._search(
            nominal_foothold=nominal,
            candidate_valid=torch.zeros(1, 3, dtype=torch.bool),
        )
        self.assertFalse(result.valid.item())
        self.assertTrue(torch.equal(result.foothold, nominal))
        self.assertEqual(result.cost.item(), 0.0)
        self.assertTrue(torch.isfinite(result.predicted_dcm).all())

    def test_positive_lateral_velocity_moves_dynamic_choice_left(self):
        candidates = torch.tensor(
            [
                [
                    [0.30, -0.08, 0.10],
                    [0.30, 0.00, 0.10],
                    [0.30, 0.08, 0.10],
                ]
            ]
        )
        result = self._search(
            candidates=candidates,
            com_velocity_xy=torch.tensor([[0.0, 0.10]]),
            desired_velocity_xy=torch.zeros(1, 2),
            weights=DCMPlannerWeights(
                nominal=0.05,
                dcm_offset=4.0,
                steepness=0.0,
                edge=0.0,
            ),
        )
        self.assertGreater(result.foothold[0, 1].item(), 0.0)

    def test_plan_is_exactly_left_right_reflection_equivariant(self):
        offsets = rectangular_search_offsets(
            [-0.04, 0.0, 0.04], [-0.08, 0.0, 0.08]
        )
        candidates = torch.zeros(2, offsets.shape[0], 3)
        candidates[:, :, :2] = torch.tensor(
            [[[0.30, 0.0]], [[0.30, 0.0]]]
        ) + offsets.unsqueeze(0)
        candidates[:, :, 2] = 0.10
        result = dcm_foothold_search(
            com_xy=torch.tensor([[0.0, 0.025], [0.0, -0.025]]),
            com_velocity_xy=torch.tensor([[0.03, 0.08], [0.03, -0.08]]),
            com_height=torch.tensor([0.70, 0.70]),
            stance_xy=torch.tensor([[0.0, -0.09], [0.0, 0.09]]),
            desired_velocity_xy=torch.tensor([[0.18, 0.0], [0.18, 0.0]]),
            nominal_foothold=torch.tensor(
                [[0.30, 0.09, 0.10], [0.30, -0.09, 0.10]]
            ),
            candidates=candidates,
            candidate_valid=torch.ones(
                2, offsets.shape[0], dtype=torch.bool
            ),
            horizon=torch.tensor([0.35, 0.35]),
        )
        self.assertAlmostEqual(
            result.foothold[0, 0].item(),
            result.foothold[1, 0].item(),
            places=6,
        )
        self.assertAlmostEqual(
            result.foothold[0, 1].item(),
            -result.foothold[1, 1].item(),
            places=6,
        )
        self.assertAlmostEqual(
            result.cost[0].item(), result.cost[1].item(), places=5
        )

    def test_extreme_horizon_and_height_remain_finite(self):
        result = self._search(
            com_height=torch.tensor([0.001]),
            horizon=torch.tensor([100.0]),
            com_velocity_xy=torch.tensor([[100.0, -100.0]]),
            weights=DCMPlannerWeights(dcm_offset=1.0),
        )
        self.assertTrue(torch.isfinite(result.foothold).all())
        self.assertTrue(torch.isfinite(result.cost).all())
        self.assertTrue(torch.isfinite(result.dcm).all())
        self.assertTrue(torch.isfinite(result.predicted_dcm).all())


if __name__ == "__main__":
    unittest.main()
