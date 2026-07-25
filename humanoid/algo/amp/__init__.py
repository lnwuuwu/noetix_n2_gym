"""AMP (Adversarial Motion Priors) 模块

提供以下组件:
- AMPDiscriminator: WGAN-GP 判别器网络
- AMPPPO: 集成 AMP 的 PPO 算法
- AMPRolloutStorage: 扩展存储, 支持 AMP 观测对
- AMPOnPolicyRunner: 集成 AMP 的训练 Runner
"""

from humanoid.algo.amp.discriminator import AMPDiscriminator
from humanoid.algo.amp.amp_ppo import AMPPPO
from humanoid.algo.amp.amp_storage import AMPRolloutStorage
from humanoid.algo.amp.amp_runner import AMPOnPolicyRunner

__all__ = [
    "AMPDiscriminator",
    "AMPPPO",
    "AMPRolloutStorage",
    "AMPOnPolicyRunner",
]
