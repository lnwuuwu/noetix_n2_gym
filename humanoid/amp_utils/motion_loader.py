import json

import torch
import numpy as np

_EPS = 1e-6


def quaternion_slerp(q0, q1, fraction, spin=0, shortestpath=True):
    """Numerically stable batched quaternion spherical interpolation."""
    del spin
    q0 = torch.nn.functional.normalize(q0, dim=-1)
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    fraction = torch.as_tensor(
        fraction, dtype=q0.dtype, device=q0.device
    )
    while fraction.ndim < q0.ndim:
        fraction = fraction.unsqueeze(-1)

    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    if shortestpath:
        q1 = torch.where(dot < 0.0, -q1, q1)
        dot = torch.abs(dot)
    dot = torch.clamp(dot, -1.0, 1.0)

    linear = torch.nn.functional.normalize(
        (1.0 - fraction) * q0 + fraction * q1, dim=-1
    )
    theta = torch.acos(torch.clamp(dot, -1.0 + _EPS, 1.0 - _EPS))
    sin_theta = torch.sin(theta)
    spherical = (
        torch.sin((1.0 - fraction) * theta) / (sin_theta + _EPS) * q0
        + torch.sin(fraction * theta) / (sin_theta + _EPS) * q1
    )
    near = torch.abs(dot) > 1.0 - 1.0e-5
    return torch.where(near, linear, spherical)

class MotionLoaderNing:

    POS_SIZE = 3
    ROT_SIZE = 4
    JOINT_POS_SIZE = 18
    TAR_TOE_POS_LOCAL_SIZE = 12
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_VEL_SIZE = 18
    BASE_HEIGHT_SIZE = 1
    MASKED_OBSERVATION_INDICES = (8, 17, 44, 53)

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE

    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE

    JOINT_POSE_START_IDX = ROOT_ROT_END_IDX
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    TAR_TOE_POS_LOCAL_START_IDX = JOINT_POSE_END_IDX
    TAR_TOE_POS_LOCAL_END_IDX = TAR_TOE_POS_LOCAL_START_IDX + TAR_TOE_POS_LOCAL_SIZE

    LINEAR_VEL_START_IDX = TAR_TOE_POS_LOCAL_END_IDX
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE

    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE

    JOINT_VEL_START_IDX = ANGULAR_VEL_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    BASE_HEIGHT_START_IDX = JOINT_VEL_END_IDX
    BASE_HEIGHT_END_IDX = BASE_HEIGHT_START_IDX + BASE_HEIGHT_SIZE


    def __init__(
            self,
            device,
            time_between_frames,
            reference_observation_horizon=2,
            num_preload_transitions=1000000,
            motion_files=""
            ):
        """Expert dataset provides AMP observations from Human mocap dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames
        self.reference_observation_horizon = reference_observation_horizon
        self.num_preload_transitions = num_preload_transitions
        self.observation_start_dim = MotionLoaderNing.JOINT_POSE_START_IDX
        
        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split('.')[0])
            with open(motion_file, "r", encoding="utf-8") as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])

                # Normalize and standardize quaternions.
                for f_i in range(motion_data.shape[0]):
                    root_rot = MotionLoaderNing.get_root_rot(motion_data[f_i])
                    motion_data[
                        f_i,
                        MotionLoaderNing.POS_SIZE:
                            (MotionLoaderNing.POS_SIZE +
                             MotionLoaderNing.ROT_SIZE)] = root_rot
                
                # Remove first 7 observation dimensions (root_pos, root_rot).
                self.trajectories.append(torch.tensor(
                    motion_data[
                        :,
                        MotionLoaderNing.ROOT_ROT_END_IDX:MotionLoaderNing.BASE_HEIGHT_END_IDX
                    ], dtype=torch.float32, device=device))
                self.trajectories_full.append(torch.tensor(
                        motion_data[:, :MotionLoaderNing.BASE_HEIGHT_END_IDX],
                        dtype=torch.float32, device=device))
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(
                    float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"Loaded {traj_len}s. motion from {motion_file}.")
        
        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        print(f'Preloading {self.num_preload_transitions} transitions')
        traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
        times = self.traj_time_sample_batch(traj_idxs)
        self.preloaded_states = torch.zeros(
            self.num_preload_transitions,
            self.reference_observation_horizon,
            self.trajectories_full[0].shape[1],
            dtype=torch.float,
            device=self.device,
            requires_grad=False
        )
        self.motion_label = torch.nn.functional.one_hot(torch.tensor(traj_idxs, dtype=torch.long, device=self.device)).float()

        for i in range(self.reference_observation_horizon):
            self.preloaded_states[:, i] = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames * i)
        print(f'Finished preloading')

        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(
            self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(
            self.trajectory_idxs, size=size, p=self.trajectory_weights,
            replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        reserved = (
            self.time_between_frames
            * (self.reference_observation_horizon - 1)
        )
        maximum_time = max(
            0.0, self.trajectory_lens[traj_idx] - reserved
        )
        return maximum_time * np.random.uniform()

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        reserved = (
            self.time_between_frames
            * (self.reference_observation_horizon - 1)
        )
        maximum_times = np.maximum(
            np.zeros_like(self.trajectory_lens[traj_idxs]),
            self.trajectory_lens[traj_idxs] - reserved,
        )
        return maximum_times * np.random.uniform(size=len(traj_idxs))

    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        frame_position = p * (n - 1)
        idx_low = int(np.floor(frame_position))
        idx_high = int(np.ceil(frame_position))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = frame_position - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        frame_positions = p * (n - 1)
        idx_low = np.floor(frame_positions).astype(np.int32)
        idx_high = np.ceil(frame_positions).astype(np.int32)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(
            frame_positions - idx_low,
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        frame_position = p * (n - 1)
        idx_low = int(np.floor(frame_position))
        idx_high = int(np.ceil(frame_position))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = frame_position - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        frame_positions = p * (n - 1)
        idx_low = np.floor(frame_positions).astype(np.int32)
        idx_high = np.ceil(frame_positions).astype(np.int32)
        all_frame_pos_starts = torch.zeros(len(traj_idxs), MotionLoaderNing.POS_SIZE, device=self.device)
        all_frame_pos_ends = torch.zeros(len(traj_idxs), MotionLoaderNing.POS_SIZE, device=self.device)
        all_frame_rot_starts = torch.zeros(len(traj_idxs), MotionLoaderNing.ROT_SIZE, device=self.device)
        all_frame_rot_ends = torch.zeros(len(traj_idxs), MotionLoaderNing.ROT_SIZE, device=self.device)
        all_frame_AMP_starts = torch.zeros(len(traj_idxs), MotionLoaderNing.BASE_HEIGHT_END_IDX - MotionLoaderNing.JOINT_POSE_START_IDX, device=self.device)
        all_frame_AMP_ends = torch.zeros(len(traj_idxs),  MotionLoaderNing.BASE_HEIGHT_END_IDX - MotionLoaderNing.JOINT_POSE_START_IDX, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask] = MotionLoaderNing.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = MotionLoaderNing.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask] = MotionLoaderNing.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = MotionLoaderNing.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            all_frame_AMP_starts[traj_mask] = trajectory[idx_low[traj_mask]][:, MotionLoaderNing.JOINT_POSE_START_IDX:MotionLoaderNing.BASE_HEIGHT_END_IDX]
            all_frame_AMP_ends[traj_mask] = trajectory[idx_high[traj_mask]][:, MotionLoaderNing.JOINT_POSE_START_IDX:MotionLoaderNing.BASE_HEIGHT_END_IDX]
        blend = torch.tensor(
            frame_positions - idx_low,
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(-1)

        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        AMP_blend = self.slerp(all_frame_AMP_starts, all_frame_AMP_ends, blend)
        return torch.cat([pos_blend, rot_blend, AMP_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        idxs = np.random.choice(
            self.preloaded_states.shape[0], size=num_frames)
        return self.preloaded_states[idxs, 0]

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """

        root_pos0, root_pos1 = MotionLoaderNing.get_root_pos(frame0), MotionLoaderNing.get_root_pos(frame1)
        root_rot0, root_rot1 = MotionLoaderNing.get_root_rot(frame0), MotionLoaderNing.get_root_rot(frame1)
        joints0, joints1 = MotionLoaderNing.get_joint_pose(frame0), MotionLoaderNing.get_joint_pose(frame1)
        tar_toe_pos_0, tar_toe_pos_1 = MotionLoaderNing.get_tar_toe_pos_local(frame0), MotionLoaderNing.get_tar_toe_pos_local(frame1)
        linear_vel_0, linear_vel_1 = MotionLoaderNing.get_linear_vel(frame0), MotionLoaderNing.get_linear_vel(frame1)
        angular_vel_0, angular_vel_1 = MotionLoaderNing.get_angular_vel(frame0), MotionLoaderNing.get_angular_vel(frame1)
        joint_vel_0, joint_vel_1 = MotionLoaderNing.get_joint_vel(frame0), MotionLoaderNing.get_joint_vel(frame1)
        base_height_0, base_height_1 = (
            MotionLoaderNing.get_base_height(frame0),
            MotionLoaderNing.get_base_height(frame1),
        )

        blend_root_pos = self.slerp(root_pos0, root_pos1, blend)
        blend_root_rot = quaternion_slerp(
            root_rot0.unsqueeze(0),
            root_rot1.unsqueeze(0),
            torch.tensor([[blend]], dtype=root_rot0.dtype, device=root_rot0.device),
        ).squeeze(0)
        blend_joints = self.slerp(joints0, joints1, blend)
        blend_tar_toe_pos = self.slerp(tar_toe_pos_0, tar_toe_pos_1, blend)
        blend_linear_vel = self.slerp(linear_vel_0, linear_vel_1, blend)
        blend_angular_vel = self.slerp(angular_vel_0, angular_vel_1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)
        blend_base_height = self.slerp(
            base_height_0, base_height_1, blend
        )

        return torch.cat([
            blend_root_pos, blend_root_rot, blend_joints, blend_tar_toe_pos,
            blend_linear_vel, blend_angular_vel, blend_joints_vel,
            blend_base_height,
        ])

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            ids = torch.randint(0, self.num_preload_transitions, (mini_batch_size,), device=self.device)
            states = self.preloaded_states[ids, :, self.observation_start_dim:]
            states[:, :, list(self.MASKED_OBSERVATION_INDICES)] = 0.0
            labels = self.motion_label[ids]
            yield states, labels

    @property
    def observation_full_dim(self):
        """Size of AMP full observations."""
        return self.trajectories_full[0].shape[1]
    
    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1]

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_root_pos(pose):
        return pose[MotionLoaderNing.ROOT_POS_START_IDX:MotionLoaderNing.ROOT_POS_END_IDX]

    def get_root_pos_batch(poses):
        return poses[:, MotionLoaderNing.ROOT_POS_START_IDX:MotionLoaderNing.ROOT_POS_END_IDX]

    def get_root_rot(pose):
        return pose[MotionLoaderNing.ROOT_ROT_START_IDX:MotionLoaderNing.ROOT_ROT_END_IDX]

    def get_root_rot_batch(poses):
        return poses[:, MotionLoaderNing.ROOT_ROT_START_IDX:MotionLoaderNing.ROOT_ROT_END_IDX]

    def get_joint_pose(pose):
        return pose[MotionLoaderNing.JOINT_POSE_START_IDX:MotionLoaderNing.JOINT_POSE_END_IDX]

    def get_joint_pose_batch(poses):
        return poses[:, MotionLoaderNing.JOINT_POSE_START_IDX:MotionLoaderNing.JOINT_POSE_END_IDX]
    
    def get_tar_toe_pos_local(pose):
        return pose[MotionLoaderNing.TAR_TOE_POS_LOCAL_START_IDX:MotionLoaderNing.TAR_TOE_POS_LOCAL_END_IDX]

    def get_tar_toe_pos_local_batch(poses):
        return poses[:, MotionLoaderNing.TAR_TOE_POS_LOCAL_START_IDX:MotionLoaderNing.TAR_TOE_POS_LOCAL_END_IDX]

    def get_linear_vel(pose):
        return pose[MotionLoaderNing.LINEAR_VEL_START_IDX:MotionLoaderNing.LINEAR_VEL_END_IDX]

    def get_linear_vel_batch(poses):
        return poses[:, MotionLoaderNing.LINEAR_VEL_START_IDX:MotionLoaderNing.LINEAR_VEL_END_IDX]

    def get_angular_vel(pose):
        return pose[MotionLoaderNing.ANGULAR_VEL_START_IDX:MotionLoaderNing.ANGULAR_VEL_END_IDX]  

    def get_angular_vel_batch(poses):
        return poses[:, MotionLoaderNing.ANGULAR_VEL_START_IDX:MotionLoaderNing.ANGULAR_VEL_END_IDX]  

    def get_joint_vel(pose):
        return pose[MotionLoaderNing.JOINT_VEL_START_IDX:MotionLoaderNing.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(poses):
        return poses[:, MotionLoaderNing.JOINT_VEL_START_IDX:MotionLoaderNing.JOINT_VEL_END_IDX]  

    def get_base_height(pose):
        return pose[
            MotionLoaderNing.BASE_HEIGHT_START_IDX:
            MotionLoaderNing.BASE_HEIGHT_END_IDX
        ]

    def get_base_height_batch(poses):
        return poses[:, MotionLoaderNing.BASE_HEIGHT_START_IDX:MotionLoaderNing.BASE_HEIGHT_END_IDX]  

class MotionLoaderNing10DOF:

    POS_SIZE = 3
    ROT_SIZE = 4
    JOINT_POS_SIZE = 10
    TAR_TOE_POS_LOCAL_SIZE = 6
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_VEL_SIZE = 10
    BASE_HEIGHT_SIZE = 1

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE

    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE

    JOINT_POSE_START_IDX = ROOT_ROT_END_IDX
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    TAR_TOE_POS_LOCAL_START_IDX = JOINT_POSE_END_IDX
    TAR_TOE_POS_LOCAL_END_IDX = TAR_TOE_POS_LOCAL_START_IDX + TAR_TOE_POS_LOCAL_SIZE

    LINEAR_VEL_START_IDX = TAR_TOE_POS_LOCAL_END_IDX
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE

    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE

    JOINT_VEL_START_IDX = ANGULAR_VEL_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    BASE_HEIGHT_START_IDX = JOINT_VEL_END_IDX
    BASE_HEIGHT_END_IDX = BASE_HEIGHT_START_IDX + BASE_HEIGHT_SIZE


    def __init__(
            self,
            device,
            time_between_frames,
            reference_observation_horizon=2,
            num_preload_transitions=1000000,
            motion_files="",
            ):
        """Expert dataset provides AMP observations from Human mocap dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames
        self.reference_observation_horizon = reference_observation_horizon
        self.num_preload_transitions = num_preload_transitions
        self.observation_start_dim = MotionLoaderNing10DOF.JOINT_POSE_START_IDX
        
        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split('.')[0])
            with open(motion_file, "r", encoding="utf-8") as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])

                # Normalize and standardize quaternions.
                for f_i in range(motion_data.shape[0]):
                    root_rot = MotionLoaderNing10DOF.get_root_rot(motion_data[f_i])
                    motion_data[
                        f_i,
                        MotionLoaderNing10DOF.POS_SIZE:
                            (MotionLoaderNing10DOF.POS_SIZE +
                             MotionLoaderNing10DOF.ROT_SIZE)] = root_rot
                
                # Remove first 7 observation dimensions (root_pos and root_orn).
                self.trajectories.append(torch.tensor(
                    motion_data[
                        :,
                        MotionLoaderNing10DOF.ROOT_ROT_END_IDX:MotionLoaderNing10DOF.BASE_HEIGHT_END_IDX
                    ], dtype=torch.float32, device=device))
                self.trajectories_full.append(torch.tensor(
                        motion_data[:, :MotionLoaderNing10DOF.BASE_HEIGHT_END_IDX],
                        dtype=torch.float32, device=device))
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(
                    float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"Loaded {traj_len}s. motion from {motion_file}.")
        
        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        print(f'Preloading {self.num_preload_transitions} transitions')
        traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
        times = self.traj_time_sample_batch(traj_idxs)
        self.preloaded_states = torch.zeros(
            self.num_preload_transitions,
            self.reference_observation_horizon,
            self.trajectories_full[0].shape[1],
            dtype=torch.float,
            device=self.device,
            requires_grad=False
        )
        self.motion_label = torch.nn.functional.one_hot(torch.tensor(traj_idxs, dtype=torch.long, device=self.device)).float()

        for i in range(self.reference_observation_horizon):
            self.preloaded_states[:, i] = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames * i)
        print(f'Finished preloading')


        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(
            self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(
            self.trajectory_idxs, size=size, p=self.trajectory_weights,
            replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = self.time_between_frames * (self.reference_observation_horizon - 1) + self.trajectory_frame_durations[traj_idx]
        return max(
            0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = self.time_between_frames * (self.reference_observation_horizon - 1) + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int32), np.ceil(p * n).astype(np.int32)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int32), np.ceil(p * n).astype(np.int32)
        all_frame_pos_starts = torch.zeros(len(traj_idxs), MotionLoaderNing10DOF.POS_SIZE, device=self.device)
        all_frame_pos_ends = torch.zeros(len(traj_idxs), MotionLoaderNing10DOF.POS_SIZE, device=self.device)
        all_frame_rot_starts = torch.zeros(len(traj_idxs), MotionLoaderNing10DOF.ROT_SIZE, device=self.device)
        all_frame_rot_ends = torch.zeros(len(traj_idxs), MotionLoaderNing10DOF.ROT_SIZE, device=self.device)
        all_frame_amp_starts = torch.zeros(len(traj_idxs), MotionLoaderNing10DOF.BASE_HEIGHT_END_IDX - MotionLoaderNing10DOF.JOINT_POSE_START_IDX, device=self.device)
        all_frame_amp_ends = torch.zeros(len(traj_idxs),  MotionLoaderNing10DOF.BASE_HEIGHT_END_IDX - MotionLoaderNing10DOF.JOINT_POSE_START_IDX, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask] = MotionLoaderNing10DOF.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = MotionLoaderNing10DOF.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask] = MotionLoaderNing10DOF.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = MotionLoaderNing10DOF.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][:, MotionLoaderNing10DOF.JOINT_POSE_START_IDX:MotionLoaderNing10DOF.BASE_HEIGHT_END_IDX]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][:, MotionLoaderNing10DOF.JOINT_POSE_START_IDX:MotionLoaderNing10DOF.BASE_HEIGHT_END_IDX]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        return torch.cat([pos_blend, rot_blend, amp_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        idxs = np.random.choice(
            self.preloaded_states.shape[0], size=num_frames)
        return self.preloaded_states[idxs, 0]

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """

        root_pos0, root_pos1 = MotionLoaderNing10DOF.get_root_pos(frame0), MotionLoaderNing10DOF.get_root_pos(frame1)
        root_rot0, root_rot1 = MotionLoaderNing10DOF.get_root_rot(frame0), MotionLoaderNing10DOF.get_root_rot(frame1)
        joints0, joints1 = MotionLoaderNing10DOF.get_joint_pose(frame0), MotionLoaderNing10DOF.get_joint_pose(frame1)
        tar_toe_pos_0, tar_toe_pos_1 = MotionLoaderNing10DOF.get_tar_toe_pos_local(frame0), MotionLoaderNing10DOF.get_tar_toe_pos_local(frame1)
        linear_vel_0, linear_vel_1 = MotionLoaderNing10DOF.get_linear_vel(frame0), MotionLoaderNing10DOF.get_linear_vel(frame1)
        angular_vel_0, angular_vel_1 = MotionLoaderNing10DOF.get_angular_vel(frame0), MotionLoaderNing10DOF.get_angular_vel(frame1)
        joint_vel_0, joint_vel_1 = MotionLoaderNing10DOF.get_joint_vel(frame0), MotionLoaderNing10DOF.get_joint_vel(frame1)

        blend_root_pos = self.slerp(root_pos0, root_pos1, blend)
        blend_root_rot = quaternion_slerp(
            root_rot0.unsqueeze(0),
            root_rot1.unsqueeze(0),
            torch.tensor([[blend]], dtype=root_rot0.dtype, device=root_rot0.device),
        ).squeeze(0)
        blend_joints = self.slerp(joints0, joints1, blend)
        blend_tar_toe_pos = self.slerp(tar_toe_pos_0, tar_toe_pos_1, blend)
        blend_linear_vel = self.slerp(linear_vel_0, linear_vel_1, blend)
        blend_angular_vel = self.slerp(angular_vel_0, angular_vel_1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat([
            blend_root_pos, blend_root_rot, blend_joints, blend_tar_toe_pos,
            blend_linear_vel, blend_angular_vel, blend_joints_vel])
    
    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            ids = torch.randint(0, self.num_preload_transitions, (mini_batch_size,), device=self.device)
            states = self.preloaded_states[ids, :, self.observation_start_dim:]
            states[:, :, [4, 9, 26, 31]] = 0.0
            labels = self.motion_label[ids]
            yield states, labels

    @property
    def observation_full_dim(self):
        """Size of AMP full observations."""
        return self.trajectories_full[0].shape[1]
    
    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1]

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_root_pos(pose):
        return pose[MotionLoaderNing10DOF.ROOT_POS_START_IDX:MotionLoaderNing10DOF.ROOT_POS_END_IDX]

    def get_root_pos_batch(poses):
        return poses[:, MotionLoaderNing10DOF.ROOT_POS_START_IDX:MotionLoaderNing10DOF.ROOT_POS_END_IDX]

    def get_root_rot(pose):
        return pose[MotionLoaderNing10DOF.ROOT_ROT_START_IDX:MotionLoaderNing10DOF.ROOT_ROT_END_IDX]

    def get_root_rot_batch(poses):
        return poses[:, MotionLoaderNing10DOF.ROOT_ROT_START_IDX:MotionLoaderNing10DOF.ROOT_ROT_END_IDX]

    def get_joint_pose(pose):
        return pose[MotionLoaderNing10DOF.JOINT_POSE_START_IDX:MotionLoaderNing10DOF.JOINT_POSE_END_IDX]

    def get_joint_pose_batch(poses):
        return poses[:, MotionLoaderNing10DOF.JOINT_POSE_START_IDX:MotionLoaderNing10DOF.JOINT_POSE_END_IDX]
    
    def get_tar_toe_pos_local(pose):
        return pose[MotionLoaderNing10DOF.TAR_TOE_POS_LOCAL_START_IDX:MotionLoaderNing10DOF.TAR_TOE_POS_LOCAL_END_IDX]

    def get_tar_toe_pos_local_batch(poses):
        return poses[:, MotionLoaderNing10DOF.TAR_TOE_POS_LOCAL_START_IDX:MotionLoaderNing10DOF.TAR_TOE_POS_LOCAL_END_IDX]

    def get_linear_vel(pose):
        return pose[MotionLoaderNing10DOF.LINEAR_VEL_START_IDX:MotionLoaderNing10DOF.LINEAR_VEL_END_IDX]

    def get_linear_vel_batch(poses):
        return poses[:, MotionLoaderNing10DOF.LINEAR_VEL_START_IDX:MotionLoaderNing10DOF.LINEAR_VEL_END_IDX]

    def get_angular_vel(pose):
        return pose[MotionLoaderNing10DOF.ANGULAR_VEL_START_IDX:MotionLoaderNing10DOF.ANGULAR_VEL_END_IDX]  

    def get_angular_vel_batch(poses):
        return poses[:, MotionLoaderNing10DOF.ANGULAR_VEL_START_IDX:MotionLoaderNing10DOF.ANGULAR_VEL_END_IDX]  

    def get_joint_vel(pose):
        return pose[MotionLoaderNing10DOF.JOINT_VEL_START_IDX:MotionLoaderNing10DOF.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(poses):
        return poses[:, MotionLoaderNing10DOF.JOINT_VEL_START_IDX:MotionLoaderNing10DOF.JOINT_VEL_END_IDX]  

class MotionLoaderNing20DOF:

    POS_SIZE = 3
    ROT_SIZE = 4
    JOINT_POS_SIZE = 20
    TAR_TOE_POS_LOCAL_SIZE = 12
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_VEL_SIZE = 20
    BASE_HEIGHT_SIZE = 1

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE

    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE

    JOINT_POSE_START_IDX = ROOT_ROT_END_IDX
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    TAR_TOE_POS_LOCAL_START_IDX = JOINT_POSE_END_IDX
    TAR_TOE_POS_LOCAL_END_IDX = TAR_TOE_POS_LOCAL_START_IDX + TAR_TOE_POS_LOCAL_SIZE

    LINEAR_VEL_START_IDX = TAR_TOE_POS_LOCAL_END_IDX
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE

    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE

    JOINT_VEL_START_IDX = ANGULAR_VEL_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    BASE_HEIGHT_START_IDX = JOINT_VEL_END_IDX
    BASE_HEIGHT_END_IDX = BASE_HEIGHT_START_IDX + BASE_HEIGHT_SIZE



    def __init__(
            self,
            device,
            time_between_frames,
            reference_observation_horizon=2,
            num_preload_transitions=1000000,
            motion_files="",
            ):
        """Expert dataset provides AMP observations from Human mocap dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames
        self.reference_observation_horizon = reference_observation_horizon
        self.num_preload_transitions = num_preload_transitions
        self.observation_start_dim = MotionLoaderNing20DOF.JOINT_POSE_START_IDX
        
        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split('.')[0])
            with open(motion_file, "r", encoding="utf-8") as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])

                # Normalize and standardize quaternions.
                for f_i in range(motion_data.shape[0]):
                    root_rot = MotionLoaderNing20DOF.get_root_rot(motion_data[f_i])
                    motion_data[
                        f_i,
                        MotionLoaderNing20DOF.POS_SIZE:
                            (MotionLoaderNing20DOF.POS_SIZE +
                             MotionLoaderNing20DOF.ROT_SIZE)] = root_rot
                
                # Remove first 7 observation dimensions (root_pos and root_orn).
                self.trajectories.append(torch.tensor(
                    motion_data[
                        :,
                        MotionLoaderNing20DOF.ROOT_ROT_END_IDX:MotionLoaderNing20DOF.BASE_HEIGHT_END_IDX
                    ], dtype=torch.float32, device=device))
                self.trajectories_full.append(torch.tensor(
                        motion_data[:, :MotionLoaderNing20DOF.BASE_HEIGHT_END_IDX],
                        dtype=torch.float32, device=device))
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(
                    float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"Loaded {traj_len}s. motion from {motion_file}.")
        
        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        print(f'Preloading {self.num_preload_transitions} transitions')
        traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
        times = self.traj_time_sample_batch(traj_idxs)
        self.preloaded_states = torch.zeros(
            self.num_preload_transitions,
            self.reference_observation_horizon,
            self.trajectories_full[0].shape[1],
            dtype=torch.float,
            device=self.device,
            requires_grad=False
        )
        self.motion_label = torch.nn.functional.one_hot(torch.tensor(traj_idxs, dtype=torch.long, device=self.device)).float()

        for i in range(self.reference_observation_horizon):
            self.preloaded_states[:, i] = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames * i)
        print(f'Finished preloading')


        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(
            self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(
            self.trajectory_idxs, size=size, p=self.trajectory_weights,
            replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = self.time_between_frames * (self.reference_observation_horizon - 1) + self.trajectory_frame_durations[traj_idx]
        return max(
            0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = self.time_between_frames * (self.reference_observation_horizon - 1) + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int32), np.ceil(p * n).astype(np.int32)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int32), np.ceil(p * n).astype(np.int32)
        all_frame_pos_starts = torch.zeros(len(traj_idxs), MotionLoaderNing20DOF.POS_SIZE, device=self.device)
        all_frame_pos_ends = torch.zeros(len(traj_idxs), MotionLoaderNing20DOF.POS_SIZE, device=self.device)
        all_frame_rot_starts = torch.zeros(len(traj_idxs), MotionLoaderNing20DOF.ROT_SIZE, device=self.device)
        all_frame_rot_ends = torch.zeros(len(traj_idxs), MotionLoaderNing20DOF.ROT_SIZE, device=self.device)
        all_frame_amp_starts = torch.zeros(len(traj_idxs), MotionLoaderNing20DOF.BASE_HEIGHT_END_IDX - MotionLoaderNing20DOF.JOINT_POSE_START_IDX, device=self.device)
        all_frame_amp_ends = torch.zeros(len(traj_idxs),  MotionLoaderNing20DOF.BASE_HEIGHT_END_IDX - MotionLoaderNing20DOF.JOINT_POSE_START_IDX, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask] = MotionLoaderNing20DOF.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = MotionLoaderNing20DOF.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask] = MotionLoaderNing20DOF.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = MotionLoaderNing20DOF.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][:, MotionLoaderNing20DOF.JOINT_POSE_START_IDX:MotionLoaderNing20DOF.BASE_HEIGHT_END_IDX]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][:, MotionLoaderNing20DOF.JOINT_POSE_START_IDX:MotionLoaderNing20DOF.BASE_HEIGHT_END_IDX]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        return torch.cat([pos_blend, rot_blend, amp_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        idxs = np.random.choice(
            self.preloaded_states.shape[0], size=num_frames)
        return self.preloaded_states[idxs, 0]

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """

        root_pos0, root_pos1 = MotionLoaderNing20DOF.get_root_pos(frame0), MotionLoaderNing20DOF.get_root_pos(frame1)
        root_rot0, root_rot1 = MotionLoaderNing20DOF.get_root_rot(frame0), MotionLoaderNing20DOF.get_root_rot(frame1)
        joints0, joints1 = MotionLoaderNing20DOF.get_joint_pose(frame0), MotionLoaderNing20DOF.get_joint_pose(frame1)
        tar_toe_pos_0, tar_toe_pos_1 = MotionLoaderNing20DOF.get_tar_toe_pos_local(frame0), MotionLoaderNing20DOF.get_tar_toe_pos_local(frame1)
        linear_vel_0, linear_vel_1 = MotionLoaderNing20DOF.get_linear_vel(frame0), MotionLoaderNing20DOF.get_linear_vel(frame1)
        angular_vel_0, angular_vel_1 = MotionLoaderNing20DOF.get_angular_vel(frame0), MotionLoaderNing20DOF.get_angular_vel(frame1)
        joint_vel_0, joint_vel_1 = MotionLoaderNing20DOF.get_joint_vel(frame0), MotionLoaderNing20DOF.get_joint_vel(frame1)

        blend_root_pos = self.slerp(root_pos0, root_pos1, blend)
        blend_root_rot = quaternion_slerp(
            root_rot0.unsqueeze(0),
            root_rot1.unsqueeze(0),
            torch.tensor([[blend]], dtype=root_rot0.dtype, device=root_rot0.device),
        ).squeeze(0)
        blend_root_rot = torch.tensor(
            motion_util.standardize_quaternion(blend_root_rot),
            dtype=torch.float32, device=self.device)
        blend_joints = self.slerp(joints0, joints1, blend)
        blend_tar_toe_pos = self.slerp(tar_toe_pos_0, tar_toe_pos_1, blend)
        blend_linear_vel = self.slerp(linear_vel_0, linear_vel_1, blend)
        blend_angular_vel = self.slerp(angular_vel_0, angular_vel_1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat([
            blend_root_pos, blend_root_rot, blend_joints, blend_tar_toe_pos,
            blend_linear_vel, blend_angular_vel, blend_joints_vel])
    
    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            ids = torch.randint(0, self.num_preload_transitions, (mini_batch_size,), device=self.device)
            states = self.preloaded_states[ids, :, self.observation_start_dim:]
            states[:, :, [8, 9, 18, 19, 46, 47, 56, 57]] = 0.0
            labels = self.motion_label[ids]
            yield states, labels

    @property
    def observation_full_dim(self):
        """Size of AMP full observations."""
        return self.trajectories_full[0].shape[1]
    
    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1]

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_root_pos(pose):
        return pose[MotionLoaderNing20DOF.ROOT_POS_START_IDX:MotionLoaderNing20DOF.ROOT_POS_END_IDX]

    def get_root_pos_batch(poses):
        return poses[:, MotionLoaderNing20DOF.ROOT_POS_START_IDX:MotionLoaderNing20DOF.ROOT_POS_END_IDX]

    def get_root_rot(pose):
        return pose[MotionLoaderNing20DOF.ROOT_ROT_START_IDX:MotionLoaderNing20DOF.ROOT_ROT_END_IDX]

    def get_root_rot_batch(poses):
        return poses[:, MotionLoaderNing20DOF.ROOT_ROT_START_IDX:MotionLoaderNing20DOF.ROOT_ROT_END_IDX]

    def get_joint_pose(pose):
        return pose[MotionLoaderNing20DOF.JOINT_POSE_START_IDX:MotionLoaderNing20DOF.JOINT_POSE_END_IDX]

    def get_joint_pose_batch(poses):
        return poses[:, MotionLoaderNing20DOF.JOINT_POSE_START_IDX:MotionLoaderNing20DOF.JOINT_POSE_END_IDX]
    
    def get_tar_toe_pos_local(pose):
        return pose[MotionLoaderNing20DOF.TAR_TOE_POS_LOCAL_START_IDX:MotionLoaderNing20DOF.TAR_TOE_POS_LOCAL_END_IDX]

    def get_tar_toe_pos_local_batch(poses):
        return poses[:, MotionLoaderNing20DOF.TAR_TOE_POS_LOCAL_START_IDX:MotionLoaderNing20DOF.TAR_TOE_POS_LOCAL_END_IDX]

    def get_linear_vel(pose):
        return pose[MotionLoaderNing20DOF.LINEAR_VEL_START_IDX:MotionLoaderNing20DOF.LINEAR_VEL_END_IDX]

    def get_linear_vel_batch(poses):
        return poses[:, MotionLoaderNing20DOF.LINEAR_VEL_START_IDX:MotionLoaderNing20DOF.LINEAR_VEL_END_IDX]

    def get_angular_vel(pose):
        return pose[MotionLoaderNing20DOF.ANGULAR_VEL_START_IDX:MotionLoaderNing20DOF.ANGULAR_VEL_END_IDX]  

    def get_angular_vel_batch(poses):
        return poses[:, MotionLoaderNing20DOF.ANGULAR_VEL_START_IDX:MotionLoaderNing20DOF.ANGULAR_VEL_END_IDX]  

    def get_joint_vel(pose):
        return pose[MotionLoaderNing20DOF.JOINT_VEL_START_IDX:MotionLoaderNing20DOF.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(poses):
        return poses[:, MotionLoaderNing20DOF.JOINT_VEL_START_IDX:MotionLoaderNing20DOF.JOINT_VEL_END_IDX]  

class MotionLoaderNingTracking:

    POS_SIZE = 3
    ROT_SIZE = 4
    JOINT_POS_SIZE = 18
    TAR_TOE_POS_LOCAL_SIZE = 63
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_VEL_SIZE = 18
    CONTACT_MASK_SIZE = 2

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE

    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE

    JOINT_POSE_START_IDX = ROOT_ROT_END_IDX
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    TAR_TOE_POS_LOCAL_START_IDX = JOINT_POSE_END_IDX
    TAR_TOE_POS_LOCAL_END_IDX = TAR_TOE_POS_LOCAL_START_IDX + TAR_TOE_POS_LOCAL_SIZE

    LINEAR_VEL_START_IDX = TAR_TOE_POS_LOCAL_END_IDX
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE

    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE

    JOINT_VEL_START_IDX = ANGULAR_VEL_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    CONTACT_MASK_START_IDX = JOINT_VEL_END_IDX
    CONTACT_MASK_END_IDX = CONTACT_MASK_START_IDX + CONTACT_MASK_SIZE


    def __init__(
            self,
            device,
            time_between_frames,
            reference_observation_horizon=2,
            num_preload_transitions=1000000,
            motion_files=""
            ):
        """Expert dataset provides AMP observations from Human mocap dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames
        self.reference_observation_horizon = reference_observation_horizon
        self.num_preload_transitions = num_preload_transitions
        self.observation_start_dim = MotionLoaderNingTracking.JOINT_POSE_START_IDX
        
        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split('.')[0])
            with open(motion_file, "r", encoding="utf-8") as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])

                # Normalize and standardize quaternions.
                for f_i in range(motion_data.shape[0]):
                    root_rot = MotionLoaderNingTracking.get_root_rot(motion_data[f_i])
                    motion_data[
                        f_i,
                        MotionLoaderNingTracking.POS_SIZE:
                            (MotionLoaderNingTracking.POS_SIZE +
                             MotionLoaderNingTracking.ROT_SIZE)] = root_rot
                
                # Remove first 7 observation dimensions (root_pos, root_rot).
                self.trajectories.append(torch.tensor(
                    motion_data[
                        :,
                        MotionLoaderNingTracking.ROOT_ROT_END_IDX:MotionLoaderNingTracking.CONTACT_MASK_END_IDX
                    ], dtype=torch.float32, device=device))
                self.trajectories_full.append(torch.tensor(
                        motion_data[:, :MotionLoaderNingTracking.CONTACT_MASK_END_IDX],
                        dtype=torch.float32, device=device))
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(
                    float(motion_json["MotionWeight"]))
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"Loaded {traj_len}s. motion from {motion_file}.")
        
        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        print(f'Preloading {self.num_preload_transitions} transitions')
        traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
        times = self.traj_time_sample_batch(traj_idxs)
        self.preloaded_states = torch.zeros(
            self.num_preload_transitions,
            self.reference_observation_horizon,
            self.trajectories_full[0].shape[1],
            dtype=torch.float,
            device=self.device,
            requires_grad=False
        )
        self.motion_label = torch.nn.functional.one_hot(torch.tensor(traj_idxs, dtype=torch.long, device=self.device)).float()

        for i in range(self.reference_observation_horizon):
            self.preloaded_states[:, i] = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames * i)
        print(f'Finished preloading')

        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(
            self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(
            self.trajectory_idxs, size=size, p=self.trajectory_weights,
            replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = self.time_between_frames * (self.reference_observation_horizon - 1) + self.trajectory_frame_durations[traj_idx]
        return max(
            0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = self.time_between_frames * (self.reference_observation_horizon - 1) + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int32), np.ceil(p * n).astype(np.int32)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int32), np.ceil(p * n).astype(np.int32)
        all_frame_pos_starts = torch.zeros(len(traj_idxs), MotionLoaderNingTracking.POS_SIZE, device=self.device)
        all_frame_pos_ends = torch.zeros(len(traj_idxs), MotionLoaderNingTracking.POS_SIZE, device=self.device)
        all_frame_rot_starts = torch.zeros(len(traj_idxs), MotionLoaderNingTracking.ROT_SIZE, device=self.device)
        all_frame_rot_ends = torch.zeros(len(traj_idxs), MotionLoaderNingTracking.ROT_SIZE, device=self.device)
        all_frame_AMP_starts = torch.zeros(len(traj_idxs), MotionLoaderNingTracking.CONTACT_MASK_END_IDX - MotionLoaderNingTracking.JOINT_POSE_START_IDX, device=self.device)
        all_frame_AMP_ends = torch.zeros(len(traj_idxs),  MotionLoaderNingTracking.CONTACT_MASK_END_IDX - MotionLoaderNingTracking.JOINT_POSE_START_IDX, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask] = MotionLoaderNingTracking.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = MotionLoaderNingTracking.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask] = MotionLoaderNingTracking.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = MotionLoaderNingTracking.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            all_frame_AMP_starts[traj_mask] = trajectory[idx_low[traj_mask]][:, MotionLoaderNingTracking.JOINT_POSE_START_IDX:MotionLoaderNingTracking.CONTACT_MASK_END_IDX]
            all_frame_AMP_ends[traj_mask] = trajectory[idx_high[traj_mask]][:, MotionLoaderNingTracking.JOINT_POSE_START_IDX:MotionLoaderNingTracking.CONTACT_MASK_END_IDX]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        AMP_blend = self.slerp(all_frame_AMP_starts, all_frame_AMP_ends, blend)
        return torch.cat([pos_blend, rot_blend, AMP_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        idxs = np.random.choice(
            self.preloaded_states.shape[0], size=num_frames)
        return self.preloaded_states[idxs, 0]

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """

        root_pos0, root_pos1 = MotionLoaderNingTracking.get_root_pos(frame0), MotionLoaderNingTracking.get_root_pos(frame1)
        root_rot0, root_rot1 = MotionLoaderNingTracking.get_root_rot(frame0), MotionLoaderNingTracking.get_root_rot(frame1)
        joints0, joints1 = MotionLoaderNingTracking.get_joint_pose(frame0), MotionLoaderNingTracking.get_joint_pose(frame1)
        tar_toe_pos_0, tar_toe_pos_1 = MotionLoaderNingTracking.get_tar_toe_pos_local(frame0), MotionLoaderNingTracking.get_tar_toe_pos_local(frame1)
        linear_vel_0, linear_vel_1 = MotionLoaderNingTracking.get_linear_vel(frame0), MotionLoaderNingTracking.get_linear_vel(frame1)
        angular_vel_0, angular_vel_1 = MotionLoaderNingTracking.get_angular_vel(frame0), MotionLoaderNingTracking.get_angular_vel(frame1)
        joint_vel_0, joint_vel_1 = MotionLoaderNingTracking.get_joint_vel(frame0), MotionLoaderNingTracking.get_joint_vel(frame1)

        blend_root_pos = self.slerp(root_pos0, root_pos1, blend)
        blend_root_rot = quaternion_slerp(
            root_rot0.unsqueeze(0),
            root_rot1.unsqueeze(0),
            torch.tensor([[blend]], dtype=root_rot0.dtype, device=root_rot0.device),
        ).squeeze(0)
        blend_joints = self.slerp(joints0, joints1, blend)
        blend_tar_toe_pos = self.slerp(tar_toe_pos_0, tar_toe_pos_1, blend)
        blend_linear_vel = self.slerp(linear_vel_0, linear_vel_1, blend)
        blend_angular_vel = self.slerp(angular_vel_0, angular_vel_1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat([
            blend_root_pos, blend_root_rot, blend_joints, blend_tar_toe_pos,
            blend_linear_vel, blend_angular_vel, blend_joints_vel])

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            ids = torch.randint(0, self.num_preload_transitions, (mini_batch_size,), device=self.device)
            states = self.preloaded_states[ids, :, self.observation_start_dim:]
            states[:, :, [8, 17, 44, 53]] = 0.0
            labels = self.motion_label[ids]
            yield states, labels

    @property
    def observation_full_dim(self):
        """Size of AMP full observations."""
        return self.trajectories_full[0].shape[1]
    
    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1]

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_root_pos(pose):
        return pose[MotionLoaderNingTracking.ROOT_POS_START_IDX:MotionLoaderNingTracking.ROOT_POS_END_IDX]

    def get_root_pos_batch(poses):
        return poses[:, MotionLoaderNingTracking.ROOT_POS_START_IDX:MotionLoaderNingTracking.ROOT_POS_END_IDX]

    def get_root_rot(pose):
        return pose[MotionLoaderNingTracking.ROOT_ROT_START_IDX:MotionLoaderNingTracking.ROOT_ROT_END_IDX]

    def get_root_rot_batch(poses):
        return poses[:, MotionLoaderNingTracking.ROOT_ROT_START_IDX:MotionLoaderNingTracking.ROOT_ROT_END_IDX]

    def get_joint_pose(pose):
        return pose[MotionLoaderNingTracking.JOINT_POSE_START_IDX:MotionLoaderNingTracking.JOINT_POSE_END_IDX]

    def get_joint_pose_batch(poses):
        return poses[:, MotionLoaderNingTracking.JOINT_POSE_START_IDX:MotionLoaderNingTracking.JOINT_POSE_END_IDX]
    
    def get_tar_toe_pos_local(pose):
        return pose[MotionLoaderNingTracking.TAR_TOE_POS_LOCAL_START_IDX:MotionLoaderNingTracking.TAR_TOE_POS_LOCAL_END_IDX]

    def get_tar_toe_pos_local_batch(poses):
        return poses[:, MotionLoaderNingTracking.TAR_TOE_POS_LOCAL_START_IDX:MotionLoaderNingTracking.TAR_TOE_POS_LOCAL_END_IDX]

    def get_linear_vel(pose):
        return pose[MotionLoaderNingTracking.LINEAR_VEL_START_IDX:MotionLoaderNingTracking.LINEAR_VEL_END_IDX]

    def get_linear_vel_batch(poses):
        return poses[:, MotionLoaderNingTracking.LINEAR_VEL_START_IDX:MotionLoaderNingTracking.LINEAR_VEL_END_IDX]

    def get_angular_vel(pose):
        return pose[MotionLoaderNingTracking.ANGULAR_VEL_START_IDX:MotionLoaderNingTracking.ANGULAR_VEL_END_IDX]  

    def get_angular_vel_batch(poses):
        return poses[:, MotionLoaderNingTracking.ANGULAR_VEL_START_IDX:MotionLoaderNingTracking.ANGULAR_VEL_END_IDX]  

    def get_joint_vel(pose):
        return pose[MotionLoaderNingTracking.JOINT_VEL_START_IDX:MotionLoaderNingTracking.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(poses):
        return poses[:, MotionLoaderNingTracking.JOINT_VEL_START_IDX:MotionLoaderNingTracking.JOINT_VEL_END_IDX]  
    
    def get_contact_mask_batch(poses):
        return poses[:, MotionLoaderNingTracking.CONTACT_MASK_START_IDX:MotionLoaderNingTracking.CONTACT_MASK_END_IDX]  
