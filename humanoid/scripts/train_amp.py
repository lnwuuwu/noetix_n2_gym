"""Train the N2 stair policy with an Adversarial Motion Prior.

This entry point deliberately reuses :mod:`humanoid.scripts.train`.  It only
resolves and validates AMP-specific inputs, selects ``AMPOnPolicyRunner`` in
the registered training config, and then lets the standard training path
create exactly one environment and one runner.
"""

from __future__ import annotations

import os
import sys

_REPOSITORY_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
if not sys.path or os.path.realpath(sys.path[0]) != _REPOSITORY_ROOT:
    sys.path.insert(0, _REPOSITORY_ROOT)

# Isaac Gym Preview 4 must load its bindings before torch or environments.
import isaacgym  # noqa: F401

from humanoid.envs import *  # noqa: F401,F403
from humanoid.scripts.train import train
from humanoid.utils.helpers import parse_humanoid_args
from humanoid.utils.task_registry import task_registry


def _standard_training_parameters():
    """CLI extensions shared with ``train.py``.

    Keeping the names and defaults identical means the standard ``train()``
    validation and post-load safeguards also apply to AMP fine-tuning.
    """
    return [
        {"name": "--fixed_terrain_level", "type": int, "default": None},
        {"name": "--terrain_level_mix", "type": str, "default": None},
        {
            "name": "--reset_optimizer",
            "action": "store_true",
            "default": False,
        },
        {"name": "--command_speed", "type": float, "default": None},
        {"name": "--learning_rate", "type": float, "default": None},
        {
            "name": "--fixed_learning_rate",
            "action": "store_true",
            "default": False,
        },
        {"name": "--action_noise_std", "type": float, "default": None},
        {"name": "--save_interval", "type": int, "default": None},
        {
            "name": "--actor_reference_loss_coeff",
            "type": float,
            "default": 0.0,
        },
        {
            "name": "--symmetrize_actor_reference",
            "action": "store_true",
            "default": False,
        },
        {
            "name": "--actor_reference_mirror_blend",
            "type": float,
            "default": 0.5,
        },
        {
            "name": "--actor_policy_loss_scale",
            "type": float,
            "default": 1.0,
        },
        {
            "name": "--actor_head_only",
            "action": "store_true",
            "default": False,
        },
        {"name": "--actor_trainable_layers", "type": int, "default": None},
        {
            "name": "--freeze_action_noise",
            "action": "store_true",
            "default": False,
        },
        {"name": "--symmetry_loss_coeff", "type": float, "default": 0.0},
        {"name": "--reward_scale_overrides", "type": str, "default": None},
        {"name": "--observation_noise_level", "type": float, "default": None},
    ]


def _amp_parameters():
    return [
        {
            "name": "--model_path",
            "type": str,
            "default": None,
            "help": "Explicit base PPO or AMP checkpoint path.",
        },
        {
            "name": "--motion_file",
            "type": str,
            "default": None,
            "help": "Comma-separated curated AMP JSON motion files.",
        },
        {
            "name": "--motion_manifest",
            "type": str,
            "default": None,
            "help": "Text file containing one curated motion path per line.",
        },
        {
            "name": "--amp_style_weight",
            "type": float,
            "default": 0.2,
        },
        {
            "name": "--amp_disc_lr",
            "type": float,
            "default": 1.0e-4,
        },
        {
            "name": "--amp_replay_size",
            "type": int,
            "default": 100000,
        },
        {
            "name": "--amp_batch_size",
            "type": int,
            "default": 512,
        },
        {
            "name": "--amp_preload_transitions",
            "type": int,
            "default": 50000,
        },
        {
            "name": "--amp_disc_updates",
            "type": int,
            "default": 2,
        },
        {
            "name": "--amp_gradient_penalty",
            "type": float,
            "default": 10.0,
        },
        {
            "name": "--amp_reward_warmup_updates",
            "type": int,
            "default": 10,
        },
        {
            "name": "--amp_reward_ramp_updates",
            "type": int,
            "default": 100,
        },
        {
            "name": "--amp_allow_legacy_motion_files",
            "action": "store_true",
            "default": False,
        },
    ]


def _absolute_existing_file(path, description, base_dir=None):
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(base_dir or os.getcwd(), path)
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError("{} does not exist: {}".format(
            description, path
        ))
    return path


def resolve_motion_files(motion_file=None, motion_manifest=None):
    """Resolve direct and manifest motion paths without silent omissions."""
    paths = []
    if motion_file:
        for path in motion_file.split(","):
            if path.strip():
                paths.append(
                    _absolute_existing_file(path.strip(), "Motion file")
                )

    if motion_manifest:
        manifest = _absolute_existing_file(
            motion_manifest, "Motion manifest"
        )
        manifest_dir = os.path.dirname(manifest)
        with open(manifest, "r") as stream:
            for line_number, line in enumerate(stream, start=1):
                entry = line.split("#", 1)[0].strip()
                if not entry:
                    continue
                try:
                    paths.append(
                        _absolute_existing_file(
                            entry, "Manifest motion", base_dir=manifest_dir
                        )
                    )
                except FileNotFoundError as error:
                    raise FileNotFoundError(
                        "{} (manifest line {})".format(error, line_number)
                    ) from error

    unique_paths = []
    seen = set()
    for path in paths:
        if path not in seen:
            unique_paths.append(path)
            seen.add(path)
    if not unique_paths:
        raise ValueError(
            "AMP needs --motion_file and/or --motion_manifest with at least "
            "one curated reference motion"
        )
    return unique_paths


def configure_amp(args):
    if args.task != "n2_stairs_walk":
        raise ValueError(
            "This curated AMP pipeline currently supports n2_stairs_walk only"
        )
    if not args.resume:
        raise ValueError(
            "AMP stair refinement is protected fine-tuning and requires "
            "--resume"
        )
    if args.model_path and (
        args.load_run is not None or args.checkpoint is not None
    ):
        raise ValueError(
            "--model_path cannot be combined with --load_run/--checkpoint"
        )
    if not args.model_path and (
        args.load_run is None and args.checkpoint is None
    ):
        raise ValueError(
            "Provide --model_path or the standard --load_run/--checkpoint "
            "resume selector"
        )
    if args.model_path:
        args.model_path = _absolute_existing_file(
            args.model_path, "Checkpoint"
        )

    motion_files = resolve_motion_files(
        args.motion_file, args.motion_manifest
    )
    _, train_cfg = task_registry.get_cfgs(args.task)
    train_cfg.runner_class_name = "AMPOnPolicyRunner"
    train_cfg.amp = {
        "motion_files": motion_files,
        "style_reward_weight": args.amp_style_weight,
        "disc_learning_rate": args.amp_disc_lr,
        "replay_buffer_size": args.amp_replay_size,
        "batch_size": args.amp_batch_size,
        "num_preload_transitions": args.amp_preload_transitions,
        "disc_updates_per_iteration": args.amp_disc_updates,
        "gradient_penalty_coefficient": args.amp_gradient_penalty,
        "reward_warmup_updates": args.amp_reward_warmup_updates,
        "reward_ramp_updates": args.amp_reward_ramp_updates,
        "allow_legacy_motion_files": bool(
            args.amp_allow_legacy_motion_files
        ),
    }
    print(
        "[AMP] Configured {} curated motion(s); checkpoint={}".format(
            len(motion_files),
            args.model_path or "standard resume selector",
        )
    )


def main():
    args = parse_humanoid_args(
        _standard_training_parameters() + _amp_parameters()
    )
    configure_amp(args)
    train(args)


if __name__ == "__main__":
    main()
