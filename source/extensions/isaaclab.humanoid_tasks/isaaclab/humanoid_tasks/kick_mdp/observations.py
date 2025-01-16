import torch
from typing import TYPE_CHECKING


from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg, ManagerTermBase, ObservationTermCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.utils.math import quat_apply_yaw

# if TYPE_CHECKING:
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def ball_rel_pos(env: ManagerBasedRLEnv, robot_asset_cfg: SceneEntityCfg, ball_asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get the ground truth state of the ball."""
    # extract the used quantities (to enable type-hinting)
    r_asset: RigidObject = env.scene[robot_asset_cfg.name]
    b_asset: RigidObject = env.scene[ball_asset_cfg.name]
    rel_pos_w = b_asset.data.root_pos_w - r_asset.data.root_pos_w
    rel_pos_r = quat_apply_yaw(r_asset.data.root_quat_w, rel_pos_w)
    return rel_pos_r

def ball_root_vel(env: ManagerBasedRLEnv, ball_asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get the ground truth state of the ball."""
    # extract the used quantities (to enable type-hinting)
    b_asset: RigidObject = env.scene[ball_asset_cfg.name]
    return b_asset.data.root_lin_vel_b

def goal_rel_pos(env: ManagerBasedRLEnv, robot_asset_cfg: SceneEntityCfg, goal_asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get the ground truth state of the goal."""
    # extract the used quantities (to enable type-hinting)
    r_asset: RigidObject = env.scene[robot_asset_cfg.name]
    g_asset: RigidObject = env.scene[goal_asset_cfg.name]
    rel_pos_w = g_asset.data.root_pos_w - r_asset.data.root_pos_w
    rel_pos_r = quat_apply_yaw(r_asset.data.root_quat_w, rel_pos_w)
    return rel_pos_r[:, :2]