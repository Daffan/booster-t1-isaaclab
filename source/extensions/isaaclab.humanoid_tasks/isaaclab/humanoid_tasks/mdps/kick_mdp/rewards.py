from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import ManagerTermBase, SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.utils.math import quat_apply_yaw

if TYPE_CHECKING:
    from omni.isaac.lab.managers import RewardTermCfg
    from isaaclab.humanoid_tasks.envs import HumanoidRLEnvCfg, HumanoidRLEnv

def approach_ball_pos(env: HumanoidRLEnv, robot_asset_cfg: SceneEntityCfg, ball_asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Encourage the robot to walk toward the ball"""
    r_asset: Articulation = env.scene[robot_asset_cfg.name]
    b_asset: RigidObject = env.scene[ball_asset_cfg.name]
    rel_pos_w = b_asset.data.root_pos_w[:, :2] - r_asset.data.root_pos_w[:, :2]
    if "rel_pos_w" in env.last_step_values:
        rel_pos_w_prev = env.last_step_values["rel_pos_w"]
        delta_distance = torch.norm(rel_pos_w, dim=-1) - torch.norm(rel_pos_w_prev, dim=-1)
    else:
        delta_distance = torch.zeros_like(torch.norm(rel_pos_w, dim=-1))
    env.last_step_values["rel_pos_w"] = rel_pos_w

    ball_root_vel = b_asset.data.root_lin_vel_b
    ball_root_vel_norm = torch.norm(ball_root_vel[:, :2], dim=-1)
    
    # only apply when the ball is static to avoid dribbling
    return -delta_distance * (ball_root_vel_norm < 0.01) * (torch.norm(rel_pos_w, dim=-1) > 0.25)

def approach_ball_lin_vel(env: HumanoidRLEnv, robot_asset_cfg: SceneEntityCfg, ball_asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Encourage the robot to walk toward the ball"""
    r_asset: Articulation = env.scene[robot_asset_cfg.name]
    b_asset: RigidObject = env.scene[ball_asset_cfg.name]
    rel_pos_w = b_asset.data.root_pos_w[:, :2] - r_asset.data.root_pos_w[:, :2]
    target_vel = rel_pos_w / torch.norm(rel_pos_w, dim=-1, keepdim=True)
    robot_vel = r_asset.data.root_lin_vel_b[:, :2]
    robot_vel_proj = torch.sum(robot_vel * target_vel, dim=-1)
    robot_vel_proj = torch.clamp(robot_vel_proj, max=1.0)  # clip to avoid large velocity

    ball_root_vel = b_asset.data.root_lin_vel_b
    ball_root_vel_norm = torch.norm(ball_root_vel[:, :2], dim=-1)

    # only apply when the ball is static to avoid dribbling
    return robot_vel_proj * (ball_root_vel_norm < 0.01) * (torch.norm(rel_pos_w, dim=-1) > 0.25)

def standstill(env: HumanoidRLEnv, robot_asset_cfg: SceneEntityCfg, ball_asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Encourage the robot to stand still"""
    r_asset: Articulation = env.scene[robot_asset_cfg.name]
    b_asset: RigidObject = env.scene[ball_asset_cfg.name]
    robot_vel = r_asset.data.root_lin_vel_b[:, :2]
    robot_vel_norm = torch.norm(robot_vel, dim=-1)

    ball_root_vel = b_asset.data.root_lin_vel_b
    ball_root_vel_norm = torch.norm(ball_root_vel, dim=-1)
    # stand still if the ball is rolling
    return -robot_vel_norm * (ball_root_vel_norm > 0.01)

def approach_ball_yaw(
        env: HumanoidRLEnv,
        robot_asset_cfg: SceneEntityCfg,
        ball_asset_cfg: SceneEntityCfg,
        std: float = 0.5,
    ) -> torch.Tensor:
    """Encourage the robot to face the ball"""
    r_asset: Articulation = env.scene[robot_asset_cfg.name]
    b_asset: RigidObject = env.scene[ball_asset_cfg.name]
    robot_root_quat = r_asset.data.root_quat_w
    # ball position in the robot frame
    ball_pos_r = quat_apply_yaw(robot_root_quat, b_asset.data.root_pos_w - r_asset.data.root_pos_w)
    ball_yaw = -torch.atan2(ball_pos_r[:, 1], ball_pos_r[:, 0])

    return torch.exp(-torch.square(ball_yaw) / std) * (torch.norm(ball_pos_r, dim=-1) > 0.25)

def ball_target(
        env: HumanoidRLEnv,
        robot_asset_cfg: SceneEntityCfg,
        ball_asset_cfg: SceneEntityCfg,
        goal_asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
    """Encourage the robot to kick the ball to the target"""
    r_asset: Articulation = env.scene[robot_asset_cfg.name]
    b_asset: RigidObject = env.scene[ball_asset_cfg.name]
    g_asset: RigidObject = env.scene[goal_asset_cfg.name]
    # command should be uniform 2D pose
    goal_pos = g_asset.data.root_pos_w[:, :2]
    rel_pos_w = goal_pos - b_asset.data.root_pos_w[:, :2]
    if "rel_ball_target_pos" in env.last_step_values:
        rel_pos_w_prev = env.last_step_values["rel_ball_target_pos"]
        delta_distance = torch.norm(rel_pos_w, dim=-1) - torch.norm(rel_pos_w_prev, dim=-1)
    else:
        delta_distance = torch.zeros_like(torch.norm(rel_pos_w, dim=-1))
    env.last_step_values["rel_ball_target_pos"] = rel_pos_w

    return -delta_distance