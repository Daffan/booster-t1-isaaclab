# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from .ppo_config import T1PPORunnerCfg

@configclass
class T1BoosterWalkROAPPORunnerCfg(T1PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.dagger_update_freq = 20

        self.experiment_name = "t1_booster_walking_roa_ppo"

        self.policy.class_name = "ActorCriticHistory"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]

        self.algorithm.class_name = "ROAPPO"
        self.algorithm.value_loss_coef = 1.0
        self.algorithm.learning_rate = 5e-4
        self.algorithm.schedule = "adaptive"
        self.algorithm.entropy_coef = 0.01
        self.algorithm.priv_reg_coef_schedual = [0, 0.1, 1000, 4000]