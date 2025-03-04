from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.humanoid_tasks.envs import SoccerRLEnv


def reset_ball_goal_pos(
    env: SoccerRLEnv,
    env_ids: torch.Tensor,
    ball_pose_range: dict[str, tuple[float, float]],
    goal_pose_range: dict[str, tuple[float, float]],
    ball_asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    goal_asset_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    minimum_distance: float = 0.3,
):
    """Reset the ball and goal root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root xy position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    ball_asset: RigidObject | Articulation = env.scene[ball_asset_cfg.name]
    goal_asset: RigidObject | Articulation = env.scene[goal_asset_cfg.name]
    # get default root state
    ball_root_states = ball_asset.data.default_root_state[env_ids].clone()
    goal_root_states = goal_asset.data.default_root_state[env_ids].clone()

    # poses
    ball_range_list = [ball_pose_range.get(key, (0.0, 0.0)) for key in ["radius"]]
    ball_angle_list = [ball_pose_range.get(key, (0.0, 2 * torch.pi)) for key in ["angle"]]
    ball_ranges = torch.tensor(ball_range_list, device=ball_asset.device)
    ball_angles = torch.tensor(ball_angle_list, device=ball_asset.device)
    goal_range_list = [goal_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y"]]
    goal_ranges = torch.tensor(goal_range_list, device=ball_asset.device)
    
    ball_radius_rand_samples = math_utils.sample_uniform(ball_ranges[:, 0], ball_ranges[:, 1], (len(env_ids), 1), device=ball_asset.device)
    ball_angle_rand_samples = math_utils.sample_uniform(ball_angles[:, 0], ball_angles[:, 1], (len(env_ids), 1), device=ball_asset.device)
    ball_rand_samples = torch.cat([ball_radius_rand_samples * torch.cos(ball_angle_rand_samples), ball_radius_rand_samples * torch.sin(ball_angle_rand_samples)], dim=-1)
    goal_rand_samples = math_utils.sample_uniform(goal_ranges[:, 0], goal_ranges[:, 1], (len(env_ids), 2), device=goal_asset.device)
    goal_rand_samples += ball_rand_samples  # make sure the goal is always in front of the ball

    # reset ball
    ball_positions = env.scene.env_origins[env_ids]
    ball_positions[:, :2] += ball_rand_samples[:, 0:2]
    # set into the physics simulation
    ball_asset.write_root_pose_to_sim(torch.cat([ball_positions, ball_root_states[:, 3:7]], dim=-1), env_ids=env_ids)
    ball_asset.write_root_velocity_to_sim(ball_root_states[:, 7:13], env_ids=env_ids)

    # reset goal
    goal_positions = env.scene.env_origins[env_ids]
    goal_positions[:, :2] += goal_rand_samples[:, 0:2]
    # set into the physics simulation
    goal_asset.write_root_pose_to_sim(torch.cat([goal_positions, goal_root_states[:, 3:7]], dim=-1), env_ids=env_ids)
    goal_asset.write_root_velocity_to_sim(goal_root_states[:, 7:13], env_ids=env_ids)