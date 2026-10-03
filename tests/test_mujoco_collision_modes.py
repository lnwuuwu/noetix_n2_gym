import importlib.util
import itertools
import unittest
from pathlib import Path

import numpy as np

from sim2sim.collision_modes import configure_robot_collision_masks


def _can_collide(contype, conaffinity, first, second):
    return bool(
        (int(contype[first]) & int(conaffinity[second]))
        or (int(contype[second]) & int(conaffinity[first])))


class MujocoCollisionMaskTest(unittest.TestCase):

    BODY_NAMES = [
        "world",
        "base_link",
        "L_leg_hip_pitch_link",
        "L_leg_knee_link",
        "R_leg_hip_pitch_link",
        "R_leg_knee_link",
    ]
    BODY_IDS = np.array([0, 1, 1, 2, 2, 3, 4, 4, 5], dtype=np.int32)
    # Every robot body has a duplicated visual 0/0 geom; the selected indices
    # below represent collision meshes with MuJoCo's default 1/1 masks.
    INITIAL_CONTYPE = np.array([1, 0, 1, 0, 1, 1, 0, 1, 1], dtype=np.int32)
    INITIAL_CONAFFINITY = INITIAL_CONTYPE.copy()
    GROUND = 0
    VISUALS = (1, 3, 6)
    TORSO = 2
    LEFT = (4, 5)
    RIGHT = (7, 8)
    ROBOT_COLLISIONS = (TORSO,) + LEFT + RIGHT

    def configured(self, mode):
        contype = self.INITIAL_CONTYPE.copy()
        conaffinity = self.INITIAL_CONAFFINITY.copy()
        configure_robot_collision_masks(
            self.BODY_IDS, contype, conaffinity, self.BODY_NAMES, mode)
        return contype, conaffinity

    def test_default_disables_all_robot_self_collision(self):
        contype, conaffinity = self.configured("disabled")
        for first, second in itertools.combinations(
                self.ROBOT_COLLISIONS, 2):
            self.assertFalse(
                _can_collide(contype, conaffinity, first, second))
        for geom in self.ROBOT_COLLISIONS:
            self.assertTrue(
                _can_collide(
                    contype, conaffinity, self.GROUND, geom),
                "robot/world contact must remain enabled")

    def test_cross_leg_enables_only_opposite_leg_pairs(self):
        contype, conaffinity = self.configured("cross_leg")
        for left, right in itertools.product(self.LEFT, self.RIGHT):
            self.assertTrue(
                _can_collide(contype, conaffinity, left, right))
        for same_leg in (self.LEFT, self.RIGHT):
            for first, second in itertools.combinations(same_leg, 2):
                self.assertFalse(
                    _can_collide(contype, conaffinity, first, second))
        for leg_geom in self.LEFT + self.RIGHT:
            self.assertFalse(
                _can_collide(
                    contype, conaffinity, self.TORSO, leg_geom))
        for geom in self.ROBOT_COLLISIONS:
            self.assertTrue(
                _can_collide(
                    contype, conaffinity, self.GROUND, geom),
                "robot/world contact must remain enabled")

    def test_visual_geoms_never_participate(self):
        all_geoms = range(len(self.BODY_IDS))
        for mode in ("disabled", "cross_leg"):
            contype, conaffinity = self.configured(mode)
            for visual in self.VISUALS:
                self.assertEqual(int(contype[visual]), 0)
                self.assertEqual(int(conaffinity[visual]), 0)
                for other in all_geoms:
                    self.assertFalse(
                        _can_collide(
                            contype, conaffinity, visual, other))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "self_collision_mode"):
            self.configured("all")

    @unittest.skipUnless(
        importlib.util.find_spec("mujoco") is not None,
        "optional mujoco package is not installed",
    )
    def test_real_n2_xml_masks_without_viewer(self):
        import mujoco

        root = Path(__file__).resolve().parents[1]
        xml_path = root / "resources/robots/N2/mjcf/N2_10dof.xml"
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        body_names = [
            mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            for body_id in range(model.nbody)]
        configure_robot_collision_masks(
            model.geom_bodyid, model.geom_contype,
            model.geom_conaffinity, body_names, "cross_leg")

        robot = model.geom_bodyid != 0
        visual = (
            robot
            & (model.geom_contype == 0)
            & (model.geom_conaffinity == 0))
        left = [
            geom_id for geom_id, body_id in enumerate(model.geom_bodyid)
            if body_names[int(body_id)].startswith("L_leg_")
            and model.geom_contype[geom_id] != 0]
        right = [
            geom_id for geom_id, body_id in enumerate(model.geom_bodyid)
            if body_names[int(body_id)].startswith("R_leg_")
            and model.geom_contype[geom_id] != 0]
        torso = [
            geom_id for geom_id, body_id in enumerate(model.geom_bodyid)
            if body_names[int(body_id)] == "base_link"
            and model.geom_contype[geom_id] != 0]

        self.assertTrue(left and right and torso)
        self.assertTrue(np.any(visual))
        for left_geom, right_geom in itertools.product(left, right):
            self.assertTrue(_can_collide(
                model.geom_contype, model.geom_conaffinity,
                left_geom, right_geom))
        for same_leg in (left, right):
            for first, second in itertools.combinations(same_leg, 2):
                self.assertFalse(_can_collide(
                    model.geom_contype, model.geom_conaffinity,
                    first, second))
        for torso_geom, leg_geom in itertools.product(
                torso, left + right):
            self.assertFalse(_can_collide(
                model.geom_contype, model.geom_conaffinity,
                torso_geom, leg_geom))


if __name__ == "__main__":
    unittest.main()
