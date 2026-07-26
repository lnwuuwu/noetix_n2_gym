"""PPO-compatible rollout storage used by AMP.

AMP transitions are kept in :class:`AMPPPO`'s dedicated replay buffer.  They
must not be appended to the PPO mini-batch tuple: ``PPO.update`` has a stable
11-item interface which is also used by recurrent policies.
"""

from humanoid.algo.ppo.rollout_storage import RolloutStorage


class AMPRolloutStorage(RolloutStorage):
    """Use the parent generators unchanged, including their 11-item output."""

    pass
