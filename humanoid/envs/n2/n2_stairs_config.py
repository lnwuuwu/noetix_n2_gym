"""Dedicated curriculum configurations for learning to climb stairs with N2."""

from humanoid.envs.n2.n2_config import N2_18DofCfg, N2_18DofCfgPPO


class N2StairsCfg(N2_18DofCfg):
    """Baseline, from-scratch upstairs task with deliberately mild randomization."""

    class init_state(N2_18DofCfg.init_state):
        # The terrain origin is a safe, flat launch point facing local +X.
        pos = [0.0, 0.0, 0.75]
        rot = [0.0, 0.0, 0.0, 1.0]
        reset_xy_noise = [0.08, 0.05]
        reset_height_offset = 0.03
        reset_velocity_noise = 0.02
        dof_position_noise = 0.04
        dof_velocity_noise = 0.10

    class env(N2_18DofCfg.env):
        num_envs = 4096
        episode_length_s = 30

        # 63 proprioceptive values + 12 forward terrain heights.
        num_single_obs = 75
        frame_stack = 5
        num_observations = frame_stack * num_single_obs

        # Critic: 63 proprio + 62 privileged dynamics/contact + 21 heights.
        num_privileged_obs = 146
        num_actions = 18
        enable_early_termination = True
        termination_height = 0.42

        # Progress watchdog. A fall, top completion, or prolonged lack of
        # forward progress ends an episode before the 30 s timeout.
        progress_grace_s = 2.0
        # Give an exploratory policy time to place a foot on the first riser;
        # 4 s caused nearly every early episode to reset at the stair face.
        stall_timeout_s = 6.0
        progress_epsilon = 0.04

        # A brief contact-filter gap is harmless, but sustained double flight
        # is a jump rather than the continuous-support gait wanted here.
        flight_grace_s = 0.5
        max_double_flight_s = 0.08

    class viewer(N2_18DofCfg.viewer):
        # A close side view of the launch platform and first stair flight.
        pos = [2.5, -4.0, 2.0]
        lookat = [2.0, 0.0, 0.7]

    class terrain(N2_18DofCfg.terrain):
        mesh_type = "trimesh"
        curriculum = True
        selected = False
        measure_heights = True
        horizontal_scale = 0.05
        vertical_scale = 0.005
        border_size = 5.0
        # The lowest riser is 0.02 / 0.05 = 0.4. Keep the trimesh conversion
        # threshold below that value so it remains a stair face, not a ramp.
        slope_treshold = 0.30

        terrain_length = 5.0
        terrain_width = 3.0
        num_rows = 5
        num_cols = 8
        max_init_terrain_level = 0
        fixed_level = -1

        # Each row is exactly one curriculum level. Every column is upstairs;
        # this proportion vector is kept explicit for audit tools and readers.
        terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        step_heights = [0.02, 0.04, 0.06, 0.08, 0.10]
        # 0.30 m is exactly representable at the 0.05 m height-field scale.
        step_width = 0.30
        num_steps = 6
        start_platform_length = 1.15
        spawn_x = 0.55
        curriculum_successes = 2
        curriculum_failures = 2
        success_height_tolerance = 0.012

        # The full 7 x 3 scan is available to the Critic. The Actor receives
        # only the forward 4 x 3 subset, keeping the deployable input compact.
        measured_points_x = [-0.20, 0.0, 0.25, 0.45, 0.65, 0.85, 1.05]
        measured_points_y = [-0.24, 0.0, 0.24]
        actor_measured_points_x = [0.25, 0.45, 0.65, 0.85]
        actor_measured_points_y = [-0.24, 0.0, 0.24]

        static_friction = 0.8
        dynamic_friction = 0.8
        restitution = 0.0

    class commands(N2_18DofCfg.commands):
        curriculum = True
        heading_command = False
        resampling_time = [1000, 1001]
        min_cmd_vel = 0.05
        initial_max_speed = 0.25
        max_curriculum = 0.45
        speed_per_terrain_level = 0.05

        class ranges:
            lin_vel_x = [0.12, 0.25]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class domain_rand(N2_18DofCfg.domain_rand):
        # Baseline: enough variation to avoid a brittle policy, but no large
        # pushes, mass shifts, or extreme contact parameters during discovery.
        action_delay = True
        action_delay_range = [0.0, 0.15]
        # Per-environment buckets are sampled at creation. Avoid thousands of
        # CPU Gym property calls on every reset during high-throughput training.
        randomize_rigid_shape_props_on_reset = False
        randomize_gains = True
        p_gain_range = [0.95, 1.05]
        d_gain_range = [0.95, 1.05]
        randomize_motor_strength = True
        motor_strength_range = [0.95, 1.05]
        randomize_com_displacement = False
        com_displacement_range = [-0.01, 0.01]
        randomize_friction = True
        friction_range = [0.70, 1.10]
        randomize_restitution = True
        restitution_range = [0.0, 0.05]
        randomize_base_mass = False
        added_mass_range = [-0.5, 0.5]
        push_robots = False
        disturbance = False

    class asset(N2_18DofCfg.asset):
        # GPU triangle-mesh net contact forces can be noisy. Constraint-only
        # foot sensors drive stance, swing, and success detection for this task.
        use_foot_force_sensors = True
        # Feet may touch risers; non-foot stair contacts are discouraged.
        penalize_contacts_on = ["hip", "knee", "shoulder", "elbow", "hand"]
        terminate_after_contacts_on = ["base"]
        # Matches the established N2 numeric indices and MJCF tree traversal;
        # N2StairsEnv validates the Isaac Gym importer result at runtime.
        expected_dof_order = [
            "L_arm_shoulder_pitch_joint",
            "L_arm_shoulder_roll_joint",
            "L_arm_shoulder_yaw_joint",
            "L_arm_elbow_joint",
            "L_leg_hip_yaw_joint",
            "L_leg_hip_roll_joint",
            "L_leg_hip_pitch_joint",
            "L_leg_knee_joint",
            "L_leg_ankle_joint",
            "R_arm_shoulder_pitch_joint",
            "R_arm_shoulder_roll_joint",
            "R_arm_shoulder_yaw_joint",
            "R_arm_elbow_joint",
            "R_leg_hip_yaw_joint",
            "R_leg_hip_roll_joint",
            "R_leg_hip_pitch_joint",
            "R_leg_knee_joint",
            "R_leg_ankle_joint",
        ]

    class rewards(N2_18DofCfg.rewards):
        soft_dof_pos_limit = 0.90
        base_height_target = 0.698
        max_contact_force = 300.0
        only_positive_rewards = False
        # The swing target follows each level: riser height + this margin.
        swing_clearance_margin = 0.06

        class scales:
            # Task progress and completion.
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.2
            stairs_forward_progress = 2.0
            # Event functions cancel the framework's dt multiplier, so these
            # are actual per-riser / terminal magnitudes rather than rates.
            stairs_foot_step_progress = 1.0
            stairs_vertical_progress = 1.0
            stairs_success = 25.0
            termination = -10.0

            # Balance, posture, and anti-cheating terms.
            orientation = 1.0
            base_height = -8.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.10
            stairs_lateral_drift = -1.0
            stairs_no_progress = -2.0
            stairs_double_flight = -4.0
            stairs_single_support = 0.80

            # Only an alternating landing with the other foot supporting can
            # receive the air-time event reward; simultaneous landings cannot.
            feet_air_time = 0.25
            stairs_swing_clearance = 0.25
            stairs_stable_contact = 0.75
            contact_no_vel = -1.0
            feet_contact_forces = -0.01
            collision = -2.0
            stumble = -1.0

            # Smooth, bounded actuation.
            default_joint_pos = 0.20
            default_up_joint_pos = 0.10
            torques = -1.0e-5
            dof_acc = -1.0e-7
            energy_cost = -2.0e-4
            action_smoothness = -0.01
            action_rate = -0.01
            dof_pos_limits = -3.0

    class noise(N2_18DofCfg.noise):
        add_noise = True
        noise_level = 0.6

        class noise_scales(N2_18DofCfg.noise.noise_scales):
            dof_pos = 0.03
            dof_vel = 0.30
            ang_vel = 0.15
            gravity = 0.03
            height_measurements = 0.015


class N2StairsCfgPPO(N2_18DofCfgPPO):
    class policy(N2_18DofCfgPPO.policy):
        class_name = "ActorCritic"
        init_noise_std = 0.8
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"

    class algorithm(N2_18DofCfgPPO.algorithm):
        class_name = "PPO"
        learning_rate = 5.0e-4
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        schedule = "adaptive"

    class runner(N2_18DofCfgPPO.runner):
        max_iterations = 8000
        num_steps_per_env = 24
        save_interval = 100
        # Random episode offsets would make the progress watchdog immediately
        # classify many freshly reset environments as stalled.
        init_at_random_ep_len = False
        experiment_name = "n2_stairs"
        run_name = "baseline_gait_v2"


class N2StairsRobustCfg(N2StairsCfg):
    """Second-stage task for fine-tuning an already capable stairs policy."""

    class domain_rand(N2StairsCfg.domain_rand):
        action_delay_range = [0.0, 0.35]
        p_gain_range = [0.85, 1.15]
        d_gain_range = [0.85, 1.15]
        motor_strength_range = [0.85, 1.15]
        randomize_com_displacement = True
        com_displacement_range = [-0.02, 0.02]
        friction_range = [0.50, 1.40]
        restitution_range = [0.0, 0.20]
        randomize_base_mass = True
        added_mass_range = [-2.0, 2.0]
        disturbance = True
        push_force_range = [20.0, 80.0]
        push_torque_range = [5.0, 25.0]
        disturbance_probabilities = 0.0005
        disturbance_interval = [10, 20]

    class terrain(N2StairsCfg.terrain):
        # Robust fine-tuning starts across all learned stair heights.
        max_init_terrain_level = 4


class N2StairsRobustCfgPPO(N2StairsCfgPPO):
    class runner(N2StairsCfgPPO.runner):
        experiment_name = "n2_stairs_robust"
        run_name = "robust"


class N2StairsWalkCfg(N2StairsCfg):
    """Human-like stair-over-stair gait with strict route and speed checks."""

    class env(N2StairsCfg.env):
        include_gait_phase = True
        include_base_lin_vel = True
        include_navigation_state = True
        enforce_walk_gait = True

        # commands(3) + phase sin/cos(2) + body velocity(3) + route state(2)
        # + legacy proprioception excluding commands(60) + terrain heights(12).
        # Explicit speed/lateral/yaw feedback lets the policy correct drift
        # instead of receiving a penalty for state it cannot observe.
        num_single_obs = 82
        num_observations = 5 * num_single_obs
        num_privileged_obs = 153
        lateral_position_obs_scale = 2.0
        yaw_error_obs_scale = 1.0

        # Phase clock is deployable: it depends only on elapsed policy time and
        # the commanded forward velocity. A short double-support interval is
        # included around every left/right transition for stair stability.
        # One half-cycle should advance one 0.30 m tread. Therefore gait
        # cycles/s = command_speed / (2 * tread_width): 0.20 Hz at 0.12 m/s
        # and 0.75 Hz at 0.45 m/s. The former 1.25 Hz clock asked the legs to
        # switch roughly four times too quickly at the 0.18 m/s test command.
        gait_frequency = 0.20
        gait_frequency_gain = 1.6666667
        gait_reference_speed = 0.12
        # Existing model_5000 checkpoints learned with the original
        # ``1.25 + 0.75 * max(command_x - 0.12, 0)`` clock. Preserve that
        # command dependence exactly at the start of fine-tuning, then blend
        # to the tread-matched target over 800 PPO iterations (24 policy
        # steps/iteration). This avoids changing the temporal meaning of the
        # phase observation at checkpoint load time.
        gait_frequency_start = 1.25
        gait_frequency_start_gain = 0.75
        gait_frequency_start_reference_speed = 0.12
        gait_frequency_transition_steps = 19200
        double_support_ratio = 0.16
        gait_reward_grace_s = 0.50
        randomize_gait_phase = True

        # A completion must settle on the top while centered, facing +X, and
        # moving close to the command. Reaching the height alone is not enough.
        top_dwell_s = 0.30
        top_speed_tolerance = 0.06
        success_max_mean_command_error = 0.06
        success_lateral_tolerance = 0.12
        success_yaw_tolerance = 0.15
        success_max_lateral_deviation = 0.20
        success_max_yaw_deviation = 0.30
        # Contact sensors on stairs do not remain perfectly phase-locked to an
        # open-loop clock: touchdown shifts with riser height. 0.70 is still
        # above the 0.58 ceiling of permanent double support, while allowing
        # adaptive touchdown timing. Repeated synchronous hopping remains
        # excluded by the independent double-flight bound.
        success_min_phase_contact_match = 0.70
        success_max_double_flight_fraction = 0.08

        # A phase match does not prove stair-over-stair gait: a policy can put
        # both feet on every tread while keeping one nominal swing leg. Count
        # actual tread indices and require alternating feet on successive
        # risers, with one missed sensor event tolerated across six steps.
        success_min_alternating_tread_count = 4
        success_min_alternating_tread_rate = 0.75
        success_max_same_tread_join_rate = 0.20
        success_max_skipped_tread_rate = 0.20

        # Prevent the visually unstable straight-leg reach seen in the first
        # strict policy. Adjacent 0.30 m treads remain comfortably reachable.
        max_sagittal_foot_offset = 0.34
        max_sagittal_foot_separation = 0.40
        overstride_soft_margin = 0.08
        success_max_sagittal_foot_separation = 0.44

        # The knee target rises mildly with riser height. N2 has no actuated
        # waist, so a small phase-locked arm swing supplies the available
        # upper-body reaction without inventing nonexistent torso joints.
        swing_knee_base_target = 0.48
        swing_knee_height_gain = 2.5
        swing_knee_max_target = 0.75
        swing_knee_tracking_sharpness = 10.0
        arm_swing_amplitude = 0.22
        arm_swing_tracking_sharpness = 12.0

        # Dense foot-order reference for stair-over-stair walking. At the
        # start of right swing, the right foot should be behind the left; it
        # crosses through zero separation and lands ahead half a cycle later.
        # The next half-cycle mirrors that motion for the left foot.
        sagittal_foot_phase_amplitude = 0.26
        sagittal_foot_phase_sharpness = 12.0
        sagittal_foot_phase_error_clip = 2.0
        next_tread_target_sharpness = 2.0
        next_tread_target_error_clip = 2.0
        # Preserve a natural early swing and introduce the absolute landing
        # target only after the foot has crossed the stance leg.
        next_tread_target_start_phase = 0.50
        next_tread_target_full_phase = 0.85

        # Leaving this center corridor is a task failure. The yaw limit is
        # deliberately looser than the success tolerance to allow recovery.
        corridor_half_width = 0.30
        corridor_yaw_limit = 0.40
        corridor_grace_s = 1.0

    class domain_rand(N2StairsCfg.domain_rand):
        # Discover the strict gait before adding actuator uncertainty. Surface
        # variation remains, while the robust task can re-enable the rest.
        action_delay = False
        randomize_gains = False
        randomize_motor_strength = False
        randomize_friction = True
        friction_range = [0.75, 1.00]
        randomize_restitution = False

    class rewards(N2StairsCfg.rewards):
        class scales(N2StairsCfg.rewards.scales):
            # Command following must dominate the former race-to-the-top
            # shortcut. Progress saturates at the commanded walking speed.
            # After the gait-clock migration, pure velocity tracking was much
            # larger than the discrete tread-sequence signal and preserved a
            # fast step-to gait. Keep command following useful without letting
            # it dominate how the six risers are negotiated.
            tracking_lin_vel = 3.0
            tracking_ang_vel = 0.5
            stairs_forward_progress = 0.50
            stairs_command_speed_error = -12.0
            stairs_overspeed = -30.0

            # Completion remains useful but no longer dominates several
            # seconds of gait, speed, and alignment penalties.
            stairs_success = 12.0

            # Explicit alternating support/swing schedule.
            stairs_phase_contact = 2.00
            stairs_phase_contact_mismatch = -3.0
            stairs_sagittal_foot_phase = 2.00
            stairs_sagittal_foot_phase_error = -0.75
            # Once stable top-reaching is established, make the late-swing
            # landing target strong enough to beat the residual step-to gait.
            # The phase ramp above still protects the natural early swing.
            stairs_next_tread_target = 3.50
            stairs_next_tread_target_error = -2.50
            stairs_double_flight = -8.0
            stairs_single_support = 0.40
            feet_air_time = 0.10
            stairs_foot_step_progress = 2.00
            stairs_alternating_tread = 5.00
            stairs_repeated_lead = -5.00
            stairs_same_tread_join = -5.00
            stairs_skipped_tread = -3.00
            stairs_stable_contact = 0.50

            # Natural joint coordination: bend the airborne knee, avoid a
            # large sagittal split, and move the arms contralaterally.
            stairs_overstride = -8.0
            stairs_swing_knee_flexion = 2.0
            # The exponential target alone has little gradient when the
            # nearly-straight legacy knee is far from its target. This
            # asymmetric deficit term supplies a usable recovery gradient.
            stairs_swing_knee_deficit = -6.0
            stairs_arm_swing = 0.50
            default_joint_pos = 0.10
            default_up_joint_pos = 0.0

            # Straight stair approach and neutral leg/foot yaw.
            stairs_lateral_drift = -8.0
            stairs_heading_alignment = 2.0
            stairs_leg_alignment = -2.0
            stairs_feet_yaw = -2.0
            lin_vel_z = -3.0

    class noise(N2StairsCfg.noise):
        noise_level = 0.4


class N2StairsWalkCfgPPO(N2StairsCfgPPO):
    class policy(N2StairsCfgPPO.policy):
        init_noise_std = 0.60

    class algorithm(N2StairsCfgPPO.algorithm):
        entropy_coef = 0.005

    class runner(N2StairsCfgPPO.runner):
        experiment_name = "n2_stairs_walk"
        run_name = "phase_walk_v1"
