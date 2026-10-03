import ast
import unittest
from pathlib import Path

import numpy as np
import yaml

from sim2sim.parkour_campaign import (
    ARENA_WIDTH,
    HORIZONTAL_SCALE,
    VERTICAL_SCALE,
    _padded_hfield_heights,
    build_campaign,
)


ROW = 3
SEED = 5
NUM_RISERS = 18
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# Deterministic row-3/seed-5 tread widths produced by the former
# U[0.20, 0.40) metre range.  The new U[0.30, 0.50) range consumes the same
# random values and must therefore add exactly one 0.1 m cell to every tread.
LEGACY_UP_TREAD_CELLS = (
    3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 3, 3, 3,
)
LEGACY_DOWN_TREAD_CELLS = (
    3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 2, 3, 3,
)


def _step_raw():
    difficulty = ROW / 8.0
    return int(
        (0.05 + difficulty * (0.20 - 0.05)) / VERTICAL_SCALE)


def _tread_cells(stage, descending=False):
    """Recover all 18 tread widths from a noiseless centreline.

    The first 17 widths are isolated constant-height runs.  The last tread is
    merged with the cropped terminal platform, so recover its width from the
    step-centre and terminal-platform goals.
    """
    profile = stage.height_field_raw[
        :, stage.height_field_raw.shape[1] // 2]
    step_raw = _step_raw()
    summit_raw = NUM_RISERS * step_raw
    if descending:
        heights = [
            summit_raw - index * step_raw
            for index in range(1, NUM_RISERS)]
    else:
        heights = [
            index * step_raw for index in range(1, NUM_RISERS)]
    widths = [int(np.count_nonzero(profile == height))
              for height in heights]

    goal_delta_cells = (
        (stage.goals[-1, 0] - stage.goals[-2, 0])
        / HORIZONTAL_SCALE)
    final_width = int(round(2.0 * (goal_delta_cells - 5.0)))
    widths.append(final_width)
    return tuple(widths)


class ParkourStairGeometryTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.legacy_campaign = build_campaign(
            row=ROW,
            seed=SEED,
            add_roughness=False,
            center_flat_goals=True,
        )
        cls.campaign = build_campaign(
            row=ROW,
            seed=SEED,
            add_roughness=False,
            center_flat_goals=True,
            step_x_range=(0.30, 0.50),
        )

    def test_default_geometry_stays_paper_compatible_and_custom_is_longer(self):
        legacy_up, legacy_down = self.legacy_campaign.stages[:2]
        self.assertEqual(_tread_cells(legacy_up), LEGACY_UP_TREAD_CELLS)
        self.assertEqual(
            _tread_cells(legacy_down, descending=True),
            LEGACY_DOWN_TREAD_CELLS,
        )
        self.assertEqual(self.legacy_campaign.step_x_range, (0.20, 0.40))
        self.assertEqual(self.campaign.step_x_range, (0.30, 0.50))

        # Use the deployed roughness setting for the public course-length
        # snapshots. Only the two stair stages grow: 18 risers x one cell x
        # two stages = 36 cells = 3.6 m.
        legacy = build_campaign(
            row=ROW, seed=SEED, add_roughness=True)
        deeper = build_campaign(
            row=ROW, seed=SEED, add_roughness=True,
            step_x_range=(0.30, 0.50))
        self.assertEqual(legacy.height_field_raw.shape, (591, 40))
        self.assertAlmostEqual(legacy.length, 59.1)
        self.assertEqual(deeper.height_field_raw.shape, (627, 40))
        self.assertAlmostEqual(deeper.length, 62.7)
        np.testing.assert_array_equal(
            np.asarray([
                stage.height_field_raw.shape[0]
                for stage in deeper.stages
            ]) - np.asarray([
                stage.height_field_raw.shape[0]
                for stage in legacy.stages
            ]),
            np.asarray([18, 18, 0, 0, 0]),
        )
        for legacy_stage, deeper_stage in zip(
                legacy.stages[2:], deeper.stages[2:]):
            np.testing.assert_array_equal(
                deeper_stage.height_field_raw,
                legacy_stage.height_field_raw,
            )
            np.testing.assert_allclose(
                deeper_stage.goals[:, 0] - deeper_stage.origin_x,
                legacy_stage.goals[:, 0] - legacy_stage.origin_x,
                rtol=0.0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                deeper_stage.goals[:, 1:],
                legacy_stage.goals[:, 1:],
                rtol=0.0,
                atol=1e-12,
            )

    def test_deployed_yaml_and_isaac_course_select_deeper_treads(self):
        yaml_path = (
            REPOSITORY_ROOT
            / "sim2sim/configs/n2_parkour_slow_stable.yaml")
        with yaml_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        self.assertEqual(config["campaign_step_x_range"], [0.30, 0.50])

        deployed = build_campaign(config)
        self.assertTrue(deployed.add_roughness)
        self.assertEqual(deployed.step_x_range, (0.30, 0.50))
        self.assertEqual(deployed.height_field_raw.shape, (627, 40))
        self.assertAlmostEqual(deployed.length, 62.7)
        np.testing.assert_allclose(
            [stage.origin_x for stage in deployed.stages]
            + [deployed.stages[-1].end_x],
            [0.0, 10.3, 20.4, 36.6, 53.3, 62.7],
            rtol=0.0,
            atol=1e-12,
        )

        # Check the actual Isaac slow-stable course class rather than only the
        # explicit build_campaign keyword used by the geometric tests.
        config_path = (
            REPOSITORY_ROOT
            / "humanoid/envs/n2/n2_parkour_config.py")
        tree = ast.parse(
            config_path.read_text(encoding="utf-8"),
            filename=str(config_path),
        )
        course_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "N2ParkourSlowStableCourseCfg")
        terrain_class = next(
            node for node in course_class.body
            if isinstance(node, ast.ClassDef) and node.name == "terrain")
        assignments = {
            target.id: ast.literal_eval(statement.value)
            for statement in terrain_class.body
            if isinstance(statement, ast.Assign)
            for target in statement.targets
            if isinstance(target, ast.Name)
        }
        self.assertEqual(
            assignments["course_step_x_range"], (0.30, 0.50))
        self.assertEqual(assignments["terrain_length"], 62.7)

    def test_each_tread_is_exactly_one_cell_deeper_than_legacy(self):
        up, down = self.campaign.stages[:2]
        expected_up = tuple(value + 1 for value in LEGACY_UP_TREAD_CELLS)
        expected_down = tuple(
            value + 1 for value in LEGACY_DOWN_TREAD_CELLS)

        actual_up = _tread_cells(up)
        actual_down = _tread_cells(down, descending=True)
        self.assertEqual(actual_up, expected_up)
        self.assertEqual(actual_down, expected_down)

        # U[0.30, 0.50) on a 0.1 m field must quantise to 0.30/0.40 m only:
        # in particular, the former 0.20 m footholds cannot survive.
        all_widths = actual_up + actual_down
        self.assertEqual(set(all_widths), {3, 4})
        self.assertNotIn(2, all_widths)

    def test_riser_height_count_and_goal_z_are_unchanged(self):
        up, down = self.campaign.stages[:2]
        mid = self.campaign.height_field_raw.shape[1] // 2
        step_raw = _step_raw()
        summit_raw = NUM_RISERS * step_raw

        up_delta = np.diff(up.height_field_raw[1:, mid].astype(np.int32))
        down_delta = np.diff(down.height_field_raw[:, mid].astype(np.int32))
        np.testing.assert_array_equal(
            up_delta[up_delta > 0],
            np.full(NUM_RISERS, step_raw, dtype=np.int32),
        )
        np.testing.assert_array_equal(
            down_delta[down_delta < 0],
            np.full(NUM_RISERS, -step_raw, dtype=np.int32),
        )
        self.assertEqual(int(up.height_field_raw[-1, mid]), summit_raw)
        self.assertEqual(int(down.height_field_raw[0, mid]), summit_raw)
        self.assertEqual(int(down.height_field_raw[-1, mid]), 0)

        goal_steps = np.asarray(
            [0, 1, 2, 3, 4, 7, 10, 14, 18, 18],
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            up.goals[:, 2],
            goal_steps * step_raw * VERTICAL_SCALE,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            down.goals[:, 2],
            (NUM_RISERS - goal_steps) * step_raw * VERTICAL_SCALE,
            rtol=0.0,
            atol=1e-12,
        )

    def test_up_down_and_flat_ground_join_in_the_policy_corridor(self):
        up, down, hurdle = self.campaign.stages[:3]
        mid = self.campaign.height_field_raw.shape[1] // 2
        # Wider than the +/-0.4 m height scan, including its interpolation
        # neighbour.  Side lowlands intentionally need not match.
        core = slice(mid - 5, mid + 6)

        np.testing.assert_array_equal(
            up.height_field_raw[-1, core],
            down.height_field_raw[0, core],
        )
        np.testing.assert_array_equal(
            down.height_field_raw[-1, core],
            hurdle.height_field_raw[0, core],
        )
        self.assertAlmostEqual(up.goals[-1, 2], down.goals[0, 2])
        self.assertAlmostEqual(down.goals[-1, 2], hurdle.goals[0, 2])

        # Along every policy-corridor column, the entire joined route may
        # change only by one unchanged riser at a time—never by a seam jump.
        joined = np.concatenate((
            up.height_field_raw[1:, core],
            down.height_field_raw[:, core],
            hurdle.height_field_raw[:1, core],
        ), axis=0).astype(np.int32)
        step_raw = _step_raw()
        for column in range(joined.shape[1]):
            delta = np.diff(joined[:, column])
            self.assertTrue(np.all(np.isin(
                delta, (-step_raw, 0, step_raw))))
            self.assertEqual(int(np.count_nonzero(delta == step_raw)),
                             NUM_RISERS)
            self.assertEqual(int(np.count_nonzero(delta == -step_raw)),
                             NUM_RISERS)

    def test_every_goal_z_matches_both_campaign_and_mujoco_hfield(self):
        campaign = self.campaign
        side_margin = 4.0
        side_cells = int(round(side_margin / HORIZONTAL_SCALE))
        mujoco_heights = _padded_hfield_heights(
            campaign,
            side_margin=side_margin,
            remove_lateral_walls=True,
            widen_stair_shoulders=True,
        )

        x_cursor = 0
        for stage in campaign.stages:
            width = stage.height_field_raw.shape[0]
            np.testing.assert_array_equal(
                campaign.height_field_raw[x_cursor:x_cursor + width],
                stage.height_field_raw,
            )
            for goal in stage.goals:
                local_x = int(np.clip(
                    (goal[0] - stage.origin_x) / HORIZONTAL_SCALE,
                    0,
                    width - 1,
                ))
                local_y = int(np.clip(
                    (goal[1] + ARENA_WIDTH / 2.0) / HORIZONTAL_SCALE,
                    0,
                    stage.height_field_raw.shape[1] - 1,
                ))
                global_x = x_cursor + local_x
                raw_z = (
                    stage.height_field_raw[local_x, local_y]
                    * VERTICAL_SCALE)
                self.assertAlmostEqual(goal[2], raw_z, places=12)
                self.assertAlmostEqual(
                    goal[2],
                    mujoco_heights[global_x, side_cells + local_y],
                    places=12,
                )
            x_cursor += width
        self.assertEqual(x_cursor, campaign.height_field_raw.shape[0])


if __name__ == "__main__":
    unittest.main()
