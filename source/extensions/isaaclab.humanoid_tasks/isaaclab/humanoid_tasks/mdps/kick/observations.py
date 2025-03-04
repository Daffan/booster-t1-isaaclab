import torch
from typing import TYPE_CHECKING


from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, ObservationTermCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_yaw, euler_xyz_from_quat, quat_rotate_inverse

from isaaclab.humanoid_tasks.envs import SoccerRLEnv

def ball_rel_pos(env: SoccerRLEnv, robot_asset_cfg: SceneEntityCfg, ball_asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get the ground truth state of the ball."""
    # extract the used quantities (to enable type-hinting)
    r_asset: RigidObject = env.scene[robot_asset_cfg.name]
    b_asset: RigidObject = env.scene[ball_asset_cfg.name]
    rel_pos_w = b_asset.data.root_pos_w - r_asset.data.root_pos_w
    rel_pos_r = quat_rotate_inverse(r_asset.data.root_quat_w, rel_pos_w)
    return rel_pos_r

def ball_rel_vel(env: SoccerRLEnv, robot_asset_cfg: SceneEntityCfg, ball_asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get the ground truth state of the ball."""
    # extract the used quantities (to enable type-hinting)
    r_asset: RigidObject = env.scene[robot_asset_cfg.name]
    b_asset: RigidObject = env.scene[ball_asset_cfg.name]
    rel_vel_r = quat_rotate_inverse(r_asset.data.root_quat_w, b_asset.data.root_lin_vel_w)
    return rel_vel_r

def goal_rel_pos(env: SoccerRLEnv, robot_asset_cfg: SceneEntityCfg, goal_asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get the ground truth state of the goal."""
    # extract the used quantities (to enable type-hinting)
    r_asset: RigidObject = env.scene[robot_asset_cfg.name]
    g_asset: RigidObject = env.scene[goal_asset_cfg.name]
    rel_pos_w = g_asset.data.root_pos_w - r_asset.data.root_pos_w
    rel_pos_r = quat_rotate_inverse(r_asset.data.root_quat_w, rel_pos_w)
    return rel_pos_r[:, :2]