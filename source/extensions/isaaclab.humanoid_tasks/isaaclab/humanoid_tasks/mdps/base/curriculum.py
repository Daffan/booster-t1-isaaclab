# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.humanoid_tasks.mdps.walk.commands import UniformVelocityFreqCommand


def modify_command_range(env: ManagerBasedRLEnv, env_ids: Sequence[int], command_term_name: str, start_scale: float, start_step: int, num_steps: int):
    """Curriculum that modifies the command range of the environment.
    """
    scale = min(1.0, max(start_scale, (1.0 - start_scale) * (env.common_step_counter - start_step) / num_steps + start_scale))

    vxm = 1.0 * scale
    vym = 1.0 * scale
    wm = 1.0 * scale

    command_term: UniformVelocityFreqCommand = env.command_manager.get_term(command_term_name)
    command_term.cfg.ranges.lin_vel_x = (-vxm, vxm)
    command_term.cfg.ranges.lin_vel_y = (-vym, vym)
    command_term.cfg.ranges.ang_vel_z = (-wm, wm)

def modify_reward_weight(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, start_step: int, num_steps: int, start_weight: float, end_weight: float):
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    step = max(env.common_step_counter - start_step, 0)
    weight = start_weight + (end_weight - start_weight) * min(step / num_steps, 1.0)

    term_cfg = env.reward_manager.get_term_cfg(term_name)
    # update term settings
    term_cfg.weight = weight
    env.reward_manager.set_term_cfg(term_name, term_cfg)