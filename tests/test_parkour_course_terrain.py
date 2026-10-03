import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sim2sim.parkour_campaign import build_campaign


REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeSubTerrain:

    def __init__(self, name, width, length, vertical_scale, horizontal_scale):
        del name
        self.width = int(width)
        self.length = int(length)
        self.vertical_scale = float(vertical_scale)
        self.horizontal_scale = float(horizontal_scale)
        self.height_field_raw = np.zeros(
            (self.width, self.length), dtype=np.int16)


def _load_terrain_module_without_isaac():
    """Load terrain.py with the tiny API surface needed by these NumPy tests."""
    terrain_utils = types.ModuleType("isaacgym.terrain_utils")
    terrain_utils.SubTerrain = _FakeSubTerrain
    isaacgym = types.ModuleType("isaacgym")
    isaacgym.terrain_utils = terrain_utils

    cfg_module = types.ModuleType(
        "humanoid.envs.base.legged_robot_config")

    class _LeggedRobotCfg:
        class terrain:
            pass

    cfg_module.LeggedRobotCfg = _LeggedRobotCfg

    replacements = {
        "isaacgym": isaacgym,
        "isaacgym.terrain_utils": terrain_utils,
        "humanoid.envs": types.ModuleType("humanoid.envs"),
        "humanoid.envs.base": types.ModuleType("humanoid.envs.base"),
        "humanoid.envs.base.legged_robot_config": cfg_module,
    }
    saved = {name: sys.modules.get(name) for name in replacements}
    try:
        sys.modules.update(replacements)
        spec = importlib.util.spec_from_file_location(
            "_parkour_course_terrain_test_module",
            REPO_ROOT / "humanoid" / "utils" / "terrain.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _course_cfg(*, center_flat_goals=False, side_margin=0.0):
    return SimpleNamespace(
        mesh_type="heightfield",
        horizontal_scale=0.1,
        vertical_scale=0.005,
        border_size=0.0,
        slope_treshold=0.75,
        curriculum=False,
        selected=False,
        terrain_kwargs=None,
        terrain_length=59.1,
        terrain_width=4.0,
        num_rows=1,
        num_cols=1,
        terrain_proportions=[0.3, 0.3, 0.2, 0.1, 0.1],
        num_goals=46,
        course_row=3,
        course_num_rows=8,
        course_seed=5,
        course_add_roughness=True,
        course_side_margin=side_margin,
        parkour_flat_center_goals=center_flat_goals,
    )


class ParkourCourseTerrainTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.terrain_module = _load_terrain_module_without_isaac()

    def test_isaac_uses_the_exact_mujoco_heightfield(self):
        expected = build_campaign(
            row=3, seed=5, add_roughness=True)
        terrain = self.terrain_module.ParkourCourseTerrain(
            _course_cfg(), num_robots=1)

        self.assertEqual(terrain.env_length, expected.length)
        self.assertEqual(terrain.env_width, expected.width)
        np.testing.assert_array_equal(
            terrain.height_field_raw, expected.height_field_raw)

    def test_goal_chain_is_g0_then_each_stages_g1_through_g9(self):
        expected = build_campaign(
            row=3, seed=5, add_roughness=True)
        terrain = self.terrain_module.ParkourCourseTerrain(
            _course_cfg(), num_robots=1)

        centred = np.concatenate(
            [expected.stages[0].goals[:1]]
            + [stage.goals[1:] for stage in expected.stages],
            axis=0,
        )
        centred[:, 1] += expected.width * 0.5
        self.assertEqual(terrain.goals.shape, (1, 1, 46, 3))
        np.testing.assert_allclose(
            terrain.goals[0, 0], centred, rtol=0.0, atol=1e-12)
        np.testing.assert_array_equal(
            terrain.course_stage_goal_starts, [1, 10, 19, 28, 37])
        np.testing.assert_array_equal(
            terrain.goal_stage_indices,
            [0] + [stage for stage in range(5) for _ in range(9)])

    def test_nominal_stairs_join_and_return_to_ground(self):
        campaign = build_campaign(
            row=3, seed=5, add_roughness=False)
        up, down, hurdle = campaign.stages[:3]
        mid = campaign.height_field_raw.shape[1] // 2

        self.assertEqual(
            int(up.height_field_raw[-1, mid]),
            int(down.height_field_raw[0, mid]))
        self.assertEqual(int(down.goals[-1, 2] / 0.005), 0)
        self.assertEqual(
            int(down.height_field_raw[-1, mid]),
            int(hurdle.height_field_raw[0, mid]))

    def test_centered_flat_course_goals_preserve_heightfield(self):
        legacy = self.terrain_module.ParkourCourseTerrain(
            _course_cfg(center_flat_goals=False), num_robots=1)
        centred = self.terrain_module.ParkourCourseTerrain(
            _course_cfg(center_flat_goals=True), num_robots=1)

        flat_start = int(centred.course_stage_goal_starts[3])
        flat_end = int(centred.course_stage_goal_starts[4])
        centre_y = centred.campaign.width * 0.5
        legacy_flat_y = legacy.goals[0, 0, flat_start:flat_end, 1]
        centred_flat_y = centred.goals[0, 0, flat_start:flat_end, 1]
        self.assertTrue(np.any(np.abs(legacy_flat_y - centre_y) > 0.0))
        np.testing.assert_array_equal(
            centred_flat_y,
            np.full_like(centred_flat_y, centre_y),
        )
        # The Isaac switch only changes targets. Terrain generation must
        # consume exactly the same random stream as the legacy course.
        np.testing.assert_array_equal(
            centred.height_field_raw,
            legacy.height_field_raw,
        )

    def test_isaac_side_shoulders_relocate_only_lateral_safety_pads(self):
        source = build_campaign(
            row=3, seed=5, add_roughness=True,
            center_flat_goals=True)
        terrain = self.terrain_module.ParkourCourseTerrain(
            _course_cfg(center_flat_goals=True, side_margin=3.0),
            num_robots=1)

        margin = 30
        source_width = source.height_field_raw.shape[1]
        self.assertEqual(terrain.height_field_raw.shape, (591, 100))
        self.assertAlmostEqual(terrain.env_width, 10.0)
        self.assertAlmostEqual(terrain.course_y_offset, 5.0)

        # The entire source interior remains byte-identical.  Only its
        # one-cell lateral pads are flattened into viewing shoulders.
        np.testing.assert_array_equal(
            terrain.height_field_raw[:, margin + 1:
                                     margin + source_width - 1],
            source.height_field_raw[:, 1:-1],
        )
        np.testing.assert_array_equal(
            terrain.height_field_raw[:, margin],
            source.height_field_raw[:, source_width // 2],
        )
        np.testing.assert_array_equal(
            terrain.height_field_raw[:, margin + source_width - 1],
            source.height_field_raw[:, source_width // 2],
        )

        # A 0.20 m collision curb follows the far shoulder height.  It keeps a
        # physical edge without recreating the source's absolute high wall.
        curb_raw = int(round(0.20 / source.vertical_scale))
        np.testing.assert_array_equal(
            terrain.height_field_raw[:, 0].astype(np.int32)
            - terrain.height_field_raw[:, 1].astype(np.int32),
            np.full(source.height_field_raw.shape[0], curb_raw),
        )
        np.testing.assert_array_equal(
            terrain.height_field_raw[:, -1].astype(np.int32)
            - terrain.height_field_raw[:, -2].astype(np.int32),
            np.full(source.height_field_raw.shape[0], curb_raw),
        )
        down = source.stages[1]
        near_down_end = int(round(
            (down.end_x - 0.2) / source.horizontal_scale))
        self.assertEqual(
            int(terrain.height_field_raw[near_down_end, margin]),
            int(source.height_field_raw[
                near_down_end, source_width // 2]),
        )

        centred = np.concatenate(
            [source.stages[0].goals[:1]]
            + [stage.goals[1:] for stage in source.stages],
            axis=0,
        )
        centred[:, 1] += 5.0
        np.testing.assert_allclose(
            terrain.goals[0, 0], centred, rtol=0.0, atol=1e-12)

    def test_default_course_is_cropped_and_deterministic(self):
        first = build_campaign(row=3, seed=5, add_roughness=True)
        second = build_campaign(row=3, seed=5, add_roughness=True)

        self.assertEqual(first.height_field_raw.shape, (591, 40))
        self.assertAlmostEqual(first.length, 59.1)
        np.testing.assert_array_equal(
            first.height_field_raw, second.height_field_raw)


if __name__ == "__main__":
    unittest.main()
