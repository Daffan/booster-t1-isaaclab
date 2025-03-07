"""
These reward functions are reimpelemented from https://github.com/BoosterRobotics/booster_gym
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.managers import RewardTermCfg
    from isaaclab.humanoid_tasks.envs import HumanoidRLEnvCfg, HumanoidRLEnv
    from isaaclab.humanoid_tasks.mdps.walk.commands import UniformVelocityFreqCommand

def tracking_lin_vel_x(
    env: HumanoidRLEnv, asset_cfg: SceneEntityCfg, std: float
):
    """Reward tracking of linear velocity commands (x axes) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term("base_velocity")
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, 0]
    filtered_lin_vel, _ = command_term.get_filtered_velocities()
    lin_vel_error = target - filtered_lin_vel[:, 0]
    lin_vel_error = torch.square(lin_vel_error)
    return torch.exp(-lin_vel_error / std)
    
def tracking_lin_vel_y(
    env: HumanoidRLEnv, asset_cfg: SceneEntityCfg, std: float
):
    """Reward tracking of linear velocity commands (y axes) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term("base_velocity")
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, 1]
    filtered_lin_vel, _ = command_term.get_filtered_velocities()
    lin_vel_error = target - filtered_lin_vel[:, 1]
    lin_vel_error = torch.square(lin_vel_error)
    return torch.exp(-lin_vel_error / std)

def tracking_ang_vel(
    env: HumanoidRLEnv, asset_cfg: SceneEntityCfg, std: float
):
    """Reward tracking of angular velocity commands using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term("base_velocity")
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, 2]
    _, filtered_ang_vel = command_term.get_filtered_velocities()
    ang_vel_error = target - filtered_ang_vel[:, 2]
    ang_vel_error = torch.square(ang_vel_error)
    return torch.exp(-ang_vel_error / std)

def survival(env: HumanoidRLEnv) -> torch.Tensor:
    """Reward for survival."""
    return torch.ones(env.num_envs, device=env.device)

def base_height(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg, target_height=0.68) -> torch.Tensor:
    """Reward for keeping the base height."""
    asset: RigidObject = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    # print("base_height", base_height[0].item())
    return torch.square(base_height - target_height)

def base_z_velocity(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for keeping the base z velocity."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_w[:, 2])

def collision(env: HumanoidRLEnv, sensor_cfg: SceneEntityCfg, threshold=0.1) -> torch.Tensor:
    """Reward for avoiding collisions."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return torch.sum(torch.norm(contact_force, dim=-1) > threshold, dim=-1)

def orientation(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for keeping the base orientation."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=-1)

def torques(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for minimizing the torques."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque), dim=-1)

def torque_tiredness(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg, threshold=0.1) -> torch.Tensor:
    """Reward for avoiding torque tiredness."""
    # TODO: not sure how to access this value
    asset: Articulation = env.scene[asset_cfg.name]
    torque_limits = asset.data.torque_limits
    torques = asset.data.applied_torque
    return torch.sum(torch.square(torques / torque_limits).clip(max=1.0), dim=-1)

def power(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for minimizing the power."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum((asset.data.applied_torque * asset.data.joint_vel).clip(min=0.0), dim=-1)

def lin_vel_z(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for keeping the linear velocity in z direction."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])

def ang_vel_xy(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for keeping the angular velocity in xy direction."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=-1)

def dof_vel(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.norm((asset.data.joint_vel), dim=1) ** 2

def dof_acc(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint accelerations on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.norm((asset.data.joint_acc), dim=1) ** 2

def root_acc(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize root accelerations."""
    asset: RigidObject = env.scene[asset_cfg.name]
    # the first body should be the Trunk, but we need to double check
    return torch.norm((asset.data.body_acc_w[:, 0]), dim=1) ** 2

def action_rate(env: HumanoidRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output"""
    return torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action), dim=1) ** 2

def dof_pos_limits(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position limits on the articulation."""
    asset: Articulation = env.scene[asset_cfg.name]
    dof_pos_limits = asset.data.soft_joint_pos_limits
    dof_pos = asset.data.joint_pos

    out_of_limits = -(dof_pos - dof_pos_limits[..., 0]).clip(max=0.) # lower limit
    out_of_limits += (dof_pos - dof_pos_limits[..., 1]).clip(min=0.)

    return torch.sum(out_of_limits, dim=1)

def feet_slip(
    env: HumanoidRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Penalize foot planar (xy) slip when in contact with the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    foot_planar_velocity = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)

    reward = is_contact * foot_planar_velocity
    return torch.sum(reward, dim=1)

def feet_yaw_diff(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the difference in yaw between the feet"""
    # make sure the asset_cfg only contains the feet joints
    asset: RigidObject = env.scene[asset_cfg.name]
    *_, feet_yaw = euler_xyz_from_quat(asset.data.body_quat_w[:, asset_cfg.body_ids].reshape(-1, 4))
    feet_yaw = feet_yaw.reshape(-1, len(asset_cfg.body_ids))
    # print("feet_yaw", wrap_to_pi(feet_yaw[:, 0] - feet_yaw[:, 1])[0].item())
    return torch.square(wrap_to_pi(feet_yaw[:, 0] - feet_yaw[:, 1]))

def feet_yaw_mean(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # make sure the asset_cfg only contains the feet joints
    asset: Articulation = env.scene[asset_cfg.name]
    *_, feet_yaw = euler_xyz_from_quat(asset.data.body_quat_w[:, asset_cfg.body_ids].reshape(-1, 4))
    feet_yaw = feet_yaw.reshape(-1, len(asset_cfg.body_ids))
    *_, base_yaw = euler_xyz_from_quat(asset.data.body_quat_w[:, 0])
    # feet_yaw_mean = feet_yaw.mean(dim=-1) + torch.pi * (torch.abs(feet_yaw[:, 1] - feet_yaw[:, 0]) > torch.pi)
    # return torch.square(wrap_to_pi(wrap_to_pi(base_yaw) - wrap_to_pi(feet_yaw_mean)))
    # print("feet_yaw_mean", (wrap_to_pi(base_yaw) - wrap_to_pi(feet_yaw.mean(dim=-1)))[0].item())
    return torch.square((wrap_to_pi(base_yaw) - wrap_to_pi(feet_yaw.mean(dim=-1))))

def feet_distance(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg, feet_distance_ref: float=0.22) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    *_, base_yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    feet_distance = torch.abs(
        torch.cos(base_yaw) * (feet_pos[:, 1, 1] - feet_pos[:, 0, 1])
        - torch.sin(base_yaw) * (feet_pos[:, 1, 0] - feet_pos[:, 0, 0])
    )
    # print("feet_distance", feet_distance[0].item())
    return torch.clip(feet_distance_ref - feet_distance, min=-0., max=0.1)
    # return torch.abs(feet_distance - feet_distance_ref)

def feet_swing(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, swing_period: float=0.2) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # get contact state
    
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 0.1

    command_term: UniformVelocityFreqCommand = env.command_manager.get_term("base_velocity")
    gait_progress = command_term.gait_progress
    gait_frequency = command_term.gait_frequency
    left_swing = (torch.abs(gait_progress - 0.25) < 0.5 * swing_period) & (gait_frequency > 1.0e-8)
    right_swing = (torch.abs(gait_progress - 0.75) < 0.5 * swing_period) & (gait_frequency > 1.0e-8)
    return (left_swing & ~is_contact[:, 0]).float() + (right_swing & ~is_contact[:, 1]).float()

def feet_swing_height(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, swing_period: float=0.2, target_height: float=0.1) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # get contact state
    
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 0.1

    command_term: UniformVelocityFreqCommand = env.command_manager.get_term("base_velocity")
    gait_progress = command_term.gait_progress
    gait_frequency = command_term.gait_frequency
    left_swing = (torch.abs(gait_progress - 0.25) < 0.5 * swing_period) & (gait_frequency > 1.0e-8)
    right_swing = (torch.abs(gait_progress - 0.75) < 0.5 * swing_period) & (gait_frequency > 1.0e-8)
    target_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] >= target_height
    return (left_swing & ~is_contact[:, 0] & target_height[:, 0]).float() + (right_swing & ~is_contact[:, 1] & target_height[:, 1]).float()

def joint_position(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command_term: UniformVelocityFreqCommand = env.command_manager.get_term("base_velocity")
    gait_frequency = command_term.gait_frequency
    # only apply when commanded to standstill
    return torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1) ** 2 * (gait_frequency < 1.0e-8)

def standstill(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    command_term: UniformVelocityFreqCommand = env.command_manager.get_term("base_velocity")
    gait_frequency = command_term.gait_frequency

    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 0.01
    # encourage both feet making contact with the ground
    return is_contact.all(dim=-1) * (gait_frequency < 1.0e-8)