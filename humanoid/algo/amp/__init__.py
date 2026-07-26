"""Adversarial Motion Prior components.

The runner is imported lazily so lightweight algorithm tests do not require
Isaac Gym or motion-file dependencies merely to import ``AMPPPO``.
"""

from humanoid.algo.amp.discriminator import AMPDiscriminator
from humanoid.algo.amp.amp_ppo import AMPPPO
from humanoid.algo.amp.amp_storage import AMPRolloutStorage

__all__ = [
    "AMPDiscriminator",
    "AMPPPO",
    "AMPRolloutStorage",
    "AMPOnPolicyRunner",
]


def __getattr__(name):
    if name == "AMPOnPolicyRunner":
        from humanoid.algo.amp.amp_runner import AMPOnPolicyRunner

        return AMPOnPolicyRunner
    raise AttributeError(name)
