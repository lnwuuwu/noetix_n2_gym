import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim2sim.parkour_campaign import (
    ARENA_WIDTH,
    Campaign,
    STAGE_COLUMNS,
    STAGE_KINDS,
    _hfield_placement,
    _padded_hfield_heights,
    build_campaign,
    build_mujoco_model,
    sample_heights,
)


class ParkourCampaignTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.campaign = build_campaign()

    def test_type_order_is_one_representative_of_play_grid(self):
        self.assertEqual(
            [stage.kind for stage in self.campaign.stages],
            list(STAGE_KINDS),
        )
        self.assertEqual(
            [stage.col for stage in self.campaign.stages],
            list(STAGE_COLUMNS),
        )
        self.assertEqual(
            [stage.label for stage in self.campaign.stages],
            ["上台阶", "下台阶", "跨栏", "平地路点", "踏石"],
        )

    def test_default_seed_replays_full_grid_rng_order(self):
        again = build_campaign()
        np.testing.assert_array_equal(
            self.campaign.height_field_raw, again.height_field_raw)
        # This snapshot also catches accidentally generating just the five
        # retained tiles instead of consuming all 8 x 10 curriculum RNG calls.
        self.assertEqual(self.campaign.height_field_raw.shape, (591, 40))
        different_seed = build_campaign(seed=6)
        self.assertFalse(np.array_equal(
            self.campaign.height_field_raw,
            different_seed.height_field_raw,
        ))

    def test_continuous_shapes_origins_and_world_goals(self):
        self.assertEqual(len(self.campaign.stages), 5)
        self.assertEqual(self.campaign.goals.shape, (50, 3))
        self.assertEqual(self.campaign.goals_by_stage.shape, (5, 10, 3))
        cursor = 0.0
        for stage in self.campaign.stages:
            self.assertEqual(stage.height_field_raw.shape[1], 40)
            self.assertEqual(stage.goals.shape, (10, 3))
            self.assertAlmostEqual(stage.origin_x, cursor)
            self.assertTrue(np.all(stage.goals[:, 0] >= stage.origin_x))
            self.assertTrue(np.all(stage.goals[:, 0] < stage.end_x))
            self.assertTrue(np.all(np.abs(stage.goals[:, 1])
                                   <= ARENA_WIDTH / 2.0))
            cursor = stage.end_x
        self.assertAlmostEqual(cursor, self.campaign.length)
        self.assertAlmostEqual(self.campaign.length, 59.1)
        self.assertAlmostEqual(self.campaign.width, 4.0)

    def test_row_three_difficulty_and_quantised_stair_offsets(self):
        campaign = build_campaign(add_roughness=False)
        self.assertEqual(campaign.row, 3)
        for stage in campaign.stages:
            self.assertAlmostEqual(stage.difficulty, 3.0 / 8.0)
        up, down = campaign.stages[:2]
        # 0.10625 m is int-quantised to 0.105 m; 18 risers -> 1.89 m.
        self.assertAlmostEqual(up.goals[-1, 2], 1.89)
        self.assertAlmostEqual(down.goals[0, 2], 1.89)
        self.assertAlmostEqual(down.height_offset, 1.89)
        self.assertAlmostEqual(down.goals[-1, 2], 0.0)

        # The centreline has no pad wall at any internal stage seam.
        for stage in campaign.stages[:-1]:
            seam = int(round(stage.end_x / campaign.horizontal_scale))
            self.assertEqual(
                campaign.height_field_raw[seam - 1, 20],
                campaign.height_field_raw[seam, 20],
            )
        self.assertTrue(np.all(campaign.height_field_raw[0] == 100))
        self.assertTrue(np.all(campaign.height_field_raw[-1] == 100))

    def test_stepping_stone_pit_is_half_a_metre_with_roughness(self):
        stones = self.campaign.stages[-1].height_field_raw
        pit = float(np.min(stones)) * self.campaign.vertical_scale
        self.assertGreaterEqual(pit, -0.55)
        self.assertLessEqual(pit, -0.45)

    def test_config_mapping_and_row_validation(self):
        campaign = build_campaign({
            "campaign": {
                "row": 4,
                "seed": 7,
                "add_roughness": False,
            }
        })
        self.assertEqual(campaign.row, 4)
        self.assertEqual(campaign.seed, 7)
        self.assertFalse(campaign.add_roughness)
        with self.assertRaisesRegex(ValueError, r"\[0, 7\]"):
            build_campaign(row=8)

    def test_centering_flat_goals_preserves_the_full_heightfield(self):
        legacy = build_campaign(
            row=3,
            seed=5,
            add_roughness=True,
            center_flat_goals=False,
        )
        centred = build_campaign(
            row=3,
            seed=5,
            add_roughness=True,
            center_flat_goals=True,
        )

        legacy_flat_y = legacy.stages[3].goals[:, 1]
        centred_flat_y = centred.stages[3].goals[:, 1]
        self.assertTrue(np.any(np.abs(legacy_flat_y) > 0.0))
        np.testing.assert_array_equal(
            centred_flat_y,
            np.zeros_like(centred_flat_y),
        )
        # Centring targets must not skip the legacy lateral RNG draws: later
        # tiles (especially stepping stones) and roughness stay byte-identical.
        np.testing.assert_array_equal(
            centred.height_field_raw,
            legacy.height_field_raw,
        )

    def test_height_sampling_uses_isaac_three_point_min(self):
        raw = np.array([
            [8, 3, 9],
            [5, 7, 9],
            [9, 9, 9],
        ], dtype=np.int16)
        campaign = Campaign(
            stages=(),
            height_field_raw=raw,
            horizontal_scale=1.0,
            vertical_scale=0.1,
            arena_length=3.0,
            arena_width=3.0,
            seed=5,
            row=3,
            add_roughness=False,
        )
        # x=0.2 -> px=0; y=-1.3 with y-origin -1.5 -> py=0.
        # min(raw[0,0], raw[1,0], raw[0,1]) = min(8,5,3) = 3.
        points = np.array([[0.2, -1.3], [99.0, 99.0]])
        sampled = sample_heights(campaign, points)
        self.assertAlmostEqual(sampled[0], 0.3)
        # Out-of-arena coordinates follow Isaac and clip to the last valid
        # triangle rather than raising.
        self.assertAlmostEqual(sampled[1], 0.7)

    def test_hfield_node_coordinates_are_not_stretched(self):
        x_radius, y_radius, center_x, center_y = _hfield_placement(
            self.campaign)
        self.assertAlmostEqual(center_x - x_radius, 0.0)
        self.assertAlmostEqual(
            center_x + x_radius,
            (self.campaign.height_field_raw.shape[0] - 1) * 0.1,
        )
        self.assertAlmostEqual(center_y - y_radius, -2.0)
        self.assertAlmostEqual(center_y + y_radius, 1.9)
        self.assertAlmostEqual(center_y, -0.05)

    def test_mujoco_side_padding_moves_skirts_without_changing_course(self):
        side_margin = 4.0
        side_nodes = int(side_margin / self.campaign.horizontal_scale)
        padded = _padded_hfield_heights(
            self.campaign, side_margin=side_margin)
        self.assertEqual(
            padded.shape,
            (
                self.campaign.height_field_raw.shape[0],
                self.campaign.height_field_raw.shape[1] + 2 * side_nodes,
            ))
        np.testing.assert_array_equal(
            padded[:, side_nodes:-side_nodes],
            self.campaign.heights)
        np.testing.assert_array_equal(
            padded[:, 0], self.campaign.heights[:, 0])
        np.testing.assert_array_equal(
            padded[:, -1], self.campaign.heights[:, -1])

        _, y_radius, _, center_y = _hfield_placement(
            self.campaign, side_margin=side_margin)
        self.assertAlmostEqual(center_y - y_radius, -6.0)
        self.assertAlmostEqual(center_y + y_radius, 5.9)
        self.assertAlmostEqual(center_y, -0.05)

    def test_mujoco_can_remove_only_lateral_pad_walls(self):
        side_margin = 4.0
        side_nodes = int(side_margin / self.campaign.horizontal_scale)
        original = self.campaign.heights
        padded = _padded_hfield_heights(
            self.campaign,
            side_margin=side_margin,
            remove_lateral_walls=True,
        )
        central = padded[:, side_nodes:-side_nodes]

        # The policy-relevant interior remains exactly Isaac-aligned.
        np.testing.assert_array_equal(central[:, 1:-1], original[:, 1:-1])
        # Only the two 0.1 m lateral pad columns become ordinary terrain.
        np.testing.assert_array_equal(central[:, 0], original[:, 1])
        np.testing.assert_array_equal(central[:, -1], original[:, -2])
        # The new solid shoulders continue those safe boundary samples.
        np.testing.assert_array_equal(padded[:, 0], original[:, 1])
        np.testing.assert_array_equal(padded[:, -1], original[:, -2])

    def test_mujoco_can_widen_stairs_while_preserving_policy_core(self):
        original = self.campaign.heights
        widened = _padded_hfield_heights(
            self.campaign,
            widen_stair_shoulders=True,
        )
        mid_y = original.shape[1] // 2
        core = slice(mid_y - 5, mid_y + 6)

        for stage in self.campaign.stages:
            start = int(round(
                stage.origin_x / self.campaign.horizontal_scale))
            stop = start + stage.height_field_raw.shape[0]
            if stage.kind in ("up_stairs", "down_stairs"):
                np.testing.assert_array_equal(
                    widened[start:stop, core],
                    original[start:stop, core],
                )
                np.testing.assert_array_equal(
                    widened[start:stop, 0],
                    original[start:stop, mid_y],
                )
                np.testing.assert_array_equal(
                    widened[start:stop, -1],
                    original[start:stop, mid_y],
                )
            else:
                np.testing.assert_array_equal(
                    widened[start:stop],
                    original[start:stop],
                )


class ParkourCampaignMujocoTest(unittest.TestCase):

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("mujoco") is not None,
        "optional mujoco package is not installed",
    )
    def test_model_replaces_boxes_and_populates_hfield(self):
        import mujoco

        xml = """\
<mujoco model="campaign_test">
  <compiler meshdir="relative_meshes"/>
  <asset/>
  <worldbody>
    <geom name="ground" type="plane" size="0 0 1"/>
    <geom name="old_stair" type="box" size=".1 1 .05"/>
  </worldbody>
</mujoco>
"""
        with tempfile.TemporaryDirectory() as directory:
            xml_path = Path(directory) / "robot.xml"
            xml_path.write_text(xml)
            campaign = build_campaign(add_roughness=False)
            side_margin = 4.0
            model = build_mujoco_model(
                str(xml_path), campaign, side_margin=side_margin)

        geom_types = list(model.geom_type)
        self.assertEqual(
            geom_types.count(mujoco.mjtGeom.mjGEOM_PLANE), 1)
        self.assertEqual(
            geom_types.count(mujoco.mjtGeom.mjGEOM_HFIELD), 1)
        self.assertEqual(
            geom_types.count(mujoco.mjtGeom.mjGEOM_BOX), 0)

        hfield_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_HFIELD, "parkour_campaign")
        self.assertEqual(
            int(model.hfield_nrow[hfield_id]),
            campaign.height_field_raw.shape[1]
            + 2 * int(side_margin / campaign.horizontal_scale),
        )
        self.assertEqual(
            int(model.hfield_ncol[hfield_id]),
            campaign.height_field_raw.shape[0],
        )
        hfield_geom = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "parkour_campaign_geom")
        x_radius, y_radius, center_x, center_y = _hfield_placement(
            campaign, side_margin=side_margin)
        self.assertAlmostEqual(model.hfield_size[hfield_id, 0], x_radius)
        self.assertAlmostEqual(model.hfield_size[hfield_id, 1], y_radius)
        self.assertAlmostEqual(model.geom_pos[hfield_geom, 0], center_x)
        self.assertAlmostEqual(model.geom_pos[hfield_geom, 1], center_y)

        ground = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.assertLess(
            model.geom_pos[ground, 2], float(campaign.heights.min()))
        address = int(model.hfield_adr[hfield_id])
        count = int(
            model.hfield_nrow[hfield_id] * model.hfield_ncol[hfield_id])
        data = model.hfield_data[address:address + count]
        self.assertAlmostEqual(float(np.min(data)), 0.0)
        self.assertAlmostEqual(float(np.max(data)), 1.0)


if __name__ == "__main__":
    unittest.main()
