from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import ManagerTermBase, SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.utils.math import quat_apply_yaw, euler_xyz_from_quat, wrap_to_pi

if TYPE_CHECKING:
    from omni.isaac.lab.managers import RewardTermCfg
    from isaaclab.humanoid_tasks.envs import HumanoidRLEnvCfg, HumanoidRLEnv

def approach_ball_pos(env: HumanoidRLEnv, robot_asset_cfg: SceneEntityCfg, ball_asset_cfg: SceneEntityCfg, distance_threshold: float=0.25) -> torch.Tensor:
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
    return -delta_distance * (ball_root_vel_norm < 0.01) * (torch.norm(rel_pos_w, dim=-1) > distance_threshold)

def approach_ball_lin_vel(
        env: HumanoidRLEnv,
        robot_asset_cfg: SceneEntityCfg,
        ball_asset_cfg: SceneEntityCfg,
        vel_clip: float=2.0,
        distance_threshold: float=0.25
    ) -> torch.Tensor:
    """Encourage the robot to walk toward the ball"""
    r_asset: Articulation = env.scene[robot_asset_cfg.name]
    b_asset: RigidObject = env.scene[ball_asset_cfg.name]
    rel_pos_w = b_asset.data.root_pos_w[:, :2] - r_asset.data.root_pos_w[:, :2]
    target_vel = rel_pos_w / torch.norm(rel_pos_w, dim=-1, keepdim=True)
    robot_vel = r_asset.data.root_lin_vel_w[:, :2]
    robot_vel_proj = torch.sum(robot_vel * target_vel, dim=-1)
    robot_vel_proj = torch.clamp(robot_vel_proj, max=vel_clip)  # clip to avoid large velocity

    ball_root_vel = b_asset.data.root_lin_vel_w
    ball_root_vel_norm = torch.norm(ball_root_vel[:, :2], dim=-1)
    # only apply when the ball is static to avoid dribbling
    return robot_vel_proj * (ball_root_vel_norm < 0.01) * (torch.norm(rel_pos_w, dim=-1) > distance_threshold)

def approach_ball_lin_vel_exp(
        env: HumanoidRLEnv,
        robot_asset_cfg: SceneEntityCfg,
        ball_asset_cfg: SceneEntityCfg,
        std: float=0.25,
        distance_threshold: float=0.25
    ) -> torch.Tensor:
    """Encourage the robot to walk toward the ball"""
    r_asset: Articulation = env.scene[robot_asset_cfg.name]
    b_asset: RigidObject = env.scene[ball_asset_cfg.name]
    command_term = env.command_manager.get_term("base_velocity")
    command = command_term.command
    lin_vel_x = command[:, 0:1]
    rel_pos_w = b_asset.data.root_pos_w[:, :2] - r_asset.data.root_pos_w[:, :2]
    target_vel = rel_pos_w / torch.norm(rel_pos_w, dim=-1, keepdim=True) * lin_vel_x
    robot_vel = r_asset.data.root_lin_vel_w[:, :2]
    vel_error = robot_vel - target_vel

    ball_root_vel = b_asset.data.root_lin_vel_w
    ball_root_vel_norm = torch.norm(ball_root_vel[:, :2], dim=-1)

    # only apply when the ball is static to avoid dribbling
    return torch.exp(-torch.square(vel_error).sum(dim=-1) / std) * (torch.norm(rel_pos_w, dim=-1) > distance_threshold) # * (ball_root_vel_norm < 0.01)

def joint_position(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg, ball_asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    b_asset: RigidObject = env.scene[ball_asset_cfg.name]
    ball_root_vel = b_asset.data.root_lin_vel_w[:, :2]
    # only apply when the ball is moving
    return torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1) ** 2 * (torch.norm(ball_root_vel, dim=-1) < 0.01)

def standstill(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg, ball_asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    b_asset: RigidObject = env.scene[ball_asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 0.01
    ball_root_vel = b_asset.data.root_lin_vel_w[:, :2]
    # encourage both feet making contact with the ground
    return is_contact.all(dim=-1) * (torch.norm(ball_root_vel, dim=-1) > 0.01)

def feet_swing(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg, ball_asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, swing_period: float=0.2) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    b_asset: RigidObject = env.scene[ball_asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # get contact state
    
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 0.1
    ball_root_vel_norm = torch.norm(b_asset.data.root_lin_vel_w[:, :2], dim=-1)

    command_term = env.command_manager.get_term("base_velocity")
    gait_progress = command_term.gait_progress
    # only swing when the ball not is moving
    left_swing = (torch.abs(gait_progress - 0.25) < 0.5 * swing_period) # & (ball_root_vel_norm < 0.01)
    right_swing = (torch.abs(gait_progress - 0.75) < 0.5 * swing_period) # & (ball_root_vel_norm < 0.01)
    return (left_swing & ~is_contact[:, 0]).float() + (right_swing & ~is_contact[:, 1]).float()

def approach_ball_yaw(
        env: HumanoidRLEnv,
        robot_asset_cfg: SceneEntityCfg,
        ball_asset_cfg: SceneEntityCfg,
        std: float = 0.5,
        distance_threshold: float=0.25
    ) -> torch.Tensor:
    """Encourage the robot to face the ball"""
    r_asset: Articulation = env.scene[robot_asset_cfg.name]
    b_asset: RigidObject = env.scene[ball_asset_cfg.name]
    robot_root_quat = r_asset.data.root_quat_w
    # ball position in the robot frame
    ball_pos_r = quat_apply_yaw(robot_root_quat, b_asset.data.root_pos_w - r_asset.data.root_pos_w)
    ball_yaw = -torch.atan2(ball_pos_r[:, 0], ball_pos_r[:, 1])
    ball_root_vel_norm = torch.norm(b_asset.data.root_lin_vel_w[:, :2], dim=-1)
    return torch.exp(-torch.square(ball_yaw) / std) * (torch.norm(ball_pos_r[:, :2], dim=-1) > distance_threshold) * (ball_root_vel_norm < 0.01)

def approach_ball_ang_vel(
        env: HumanoidRLEnv,
        robot_asset_cfg: SceneEntityCfg,
        ball_asset_cfg: SceneEntityCfg,
        std: float = 0.5,
        distance_threshold: float=0.25
    ) -> torch.Tensor:
    """Encourage the robot to face the ball"""
    r_asset: Articulation = env.scene[robot_asset_cfg.name]
    b_asset: RigidObject = env.scene[ball_asset_cfg.name]
    # ball position in the robot frame
    ball_pos_w = b_asset.data.root_pos_w - r_asset.data.root_pos_w
    ball_yaw = torch.atan2(ball_pos_w[:, 1], ball_pos_w[:, 0])
    flipped_ball_yaw1 = ball_yaw + 2 * torch.pi
    flipped_ball_yaw2 = ball_yaw - 2 * torch.pi
    ball_root_vel_norm = torch.norm(b_asset.data.root_lin_vel_w[:, :2], dim=-1)

    ball_yaw_all = torch.stack([ball_yaw, flipped_ball_yaw1, flipped_ball_yaw2], dim=-1)
    yaw_diff_select = torch.argmin(torch.abs(ball_yaw_all - r_asset.data.heading_w.unsqueeze(-1)), dim=-1)
    yaw_diff = (ball_yaw_all - r_asset.data.heading_w.unsqueeze(-1)).gather(-1, yaw_diff_select.unsqueeze(-1)).squeeze(-1)

    command_term = env.command_manager.get_term("base_velocity")
    command = command_term.command
    yaw_speed = command[:, 2]

    yaw_speed_target = yaw_diff.sign() * yaw_speed * (yaw_diff.abs() > 0.1).float()
    yaw_speed_curr = r_asset.data.root_ang_vel_w[:, 2]
    yaw_speed_error = yaw_speed_target - yaw_speed_curr
    
    # if torch.abs(yaw_diff).item() < 0.1:
    #     import ipdb; ipdb.set_trace()

    # print(yaw_diff.item(), ball_yaw.item(), yaw_speed_target.item(), yaw_speed_curr.item(), yaw_speed_error.item())
    # print(torch.exp(-torch.square(yaw_speed_error) / std))
    return torch.exp(-torch.square(yaw_speed_error) / std) * (torch.norm(ball_pos_w[:, :2], dim=-1) > distance_threshold) # * (ball_root_vel_norm < 0.01)

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

    # change of ball position is not discouraged
    return torch.clamp(-delta_distance, min=0)

def ball_velocity(
        env: HumanoidRLEnv,
        robot_asset_cfg: SceneEntityCfg,
        ball_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    b_asset: RigidObject = env.scene[ball_asset_cfg.name]
    ball_root_vel = b_asset.data.root_lin_vel_b
    ball_root_vel_norm = torch.norm(ball_root_vel[:, :2], dim=-1)
    return ball_root_vel_norm
