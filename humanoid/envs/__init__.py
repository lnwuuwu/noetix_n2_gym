from humanoid import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from humanoid.utils.task_registry import task_registry

# ---------------------------------------------- Base ----------------------------------------------
from .n2.n2_env import N2Env
from .n2.n2_config import N2_18DofCfg, N2_18DofCfgPPO
from .n2.n2_10dof_env import N2_10dof_Env
from .n2.n2_10dof_config import N2_10dof_Cfg, N2_10dof_CfgPPO
from .n2.n2_perceptive_env import N2PerceptiveEnv
from .n2.n2_perceptive_config import N2PerceptiveCfg, N2PerceptiveCfgPPO

task_registry.register( "n2", N2Env, N2_18DofCfg(), N2_18DofCfgPPO() )
task_registry.register( "n2_10dof", N2_10dof_Env, N2_10dof_Cfg(), N2_10dof_CfgPPO() )
task_registry.register('n2_perceptive', N2PerceptiveEnv, N2PerceptiveCfg(), N2PerceptiveCfgPPO())

# ---------------------------------------------- Parkour ----------------------------------------------
from .n2.n2_parkour_env import N2ParkourEnv
from .n2.n2_parkour_config import (
    N2ParkourCfg,
    N2ParkourCfgPPO,
    N2ParkourCourseCfg,
    N2ParkourCourseCfgPPO,
    N2ParkourSlowStableCfg,
    N2ParkourSlowStableCfgPPO,
    N2ParkourSlowStableCourseCfg,
    N2ParkourSlowStableCourseCfgPPO,
    N2ParkourStabilityCfg,
    N2ParkourStabilityCfgPPO,
)
task_registry.register('n2_parkour', N2ParkourEnv, N2ParkourCfg(), N2ParkourCfgPPO())
task_registry.register(
    'n2_parkour_course',
    N2ParkourEnv,
    N2ParkourCourseCfg(),
    N2ParkourCourseCfgPPO())
task_registry.register(
    'n2_parkour_stability',
    N2ParkourEnv,
    N2ParkourStabilityCfg(),
    N2ParkourStabilityCfgPPO())
task_registry.register(
    'n2_parkour_slow_stable',
    N2ParkourEnv,
    N2ParkourSlowStableCfg(),
    N2ParkourSlowStableCfgPPO())
task_registry.register(
    'n2_parkour_slow_stable_course',
    N2ParkourEnv,
    N2ParkourSlowStableCourseCfg(),
    N2ParkourSlowStableCourseCfgPPO())

# ---------------------------------------------- Mimic ----------------------------------------------
from .n2.n2_mimic_env import N2MimicEnv
from .n2.n2_mimic_config import N2MimicCfg, N2MimicCfgPPO
task_registry.register( "n2_mimic", N2MimicEnv, N2MimicCfg(), N2MimicCfgPPO() )





