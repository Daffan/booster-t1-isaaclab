from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.managers import RewardTermCfg
    from isaaclab.humanoid_tasks.envs import HumanoidRLEnvCfg, HumanoidRLEnv


###############  Task Reward Terms (old)  ###############

def survival_reward(env: HumanoidRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)

def base_angular_velocity_reward(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, 2]
    ang_vel_error = torch.linalg.norm((target - asset.data.root_ang_vel_b[:, 2]).unsqueeze(1), dim=1) ** 2
    return torch.exp(-ang_vel_error / std)

def base_linear_velocity_reward(
    env: HumanoidRLEnv, asset_cfg: SceneEntityCfg, std: float, ramp_at_vel: float = 1.0, ramp_rate: float = 0.5
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, :2]
    lin_vel_error = torch.square((target - asset.data.root_lin_vel_b[:, :2]))
    # fixed 1.0 multiple for tracking below the ramp_at_vel value, then scale by the rate above
    vel_cmd_magnitude = torch.linalg.norm(target, dim=1)
    velocity_scaling_multiple = torch.clamp(1.0 + ramp_rate * (vel_cmd_magnitude - ramp_at_vel), min=1.0)
    return torch.sum(torch.exp(-lin_vel_error / std), dim=1) * velocity_scaling_multiple


###############  Task Reward Terms (new) ###############

def tracking_lin_vel_reward(
    env: HumanoidRLEnv, asset_cfg: SceneEntityCfg, std: float
):
    """Reward tracking of linear velocity commands (xy axes) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, :2]
    lin_vel_error = target - asset.data.root_lin_vel_b[:, :2]
    lin_vel_error *= 1. / (1. + torch.abs(target[:, :2]))  # scale by torque magnitude
    lin_vel_error = torch.sum(torch.square(lin_vel_error), dim=1)
    return torch.exp(-lin_vel_error / std)

def tracking_ang_vel_reward(
    env: HumanoidRLEnv, asset_cfg: SceneEntityCfg, std: float):
    # Reward tracking yaw angular velocity command
    asset: RigidObject = env.scene[asset_cfg.name]
    target = env.command_manager.get_command("base_velocity")[:, 2]
    ang_vel_error = torch.square(
        (target - asset.data.root_ang_vel_b[:, 2]) * 2 / torch.pi)
    return torch.exp(-ang_vel_error / std)

def base_height_reward(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg, target_height: float=0.68, std: float=0.25) -> torch.Tensor:
    """Reward tracking of base height commands using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    base_height_error = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    return torch.exp(-base_height_error / std)

def base_orientation_penalty(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize non-flat base orientation
    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.projected_gravity_b[:, :2]), dim=1)

###############  Auxiliary Reward Terms  ###############

def action_rate1_reward(env: HumanoidRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output"""
    return torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action) / env.step_dt, dim=1) ** 2
    # return torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action), dim=1)

def action_rate2_reward(env: HumanoidRLEnv) -> torch.Tensor:
    """Penalize large second-order derivative in the network action output"""
    if len(env.action_histories) < 3:
        return torch.zeros(env.num_envs, device=env.device)
    else:
        diff = env.action_histories[0] - 2 * env.action_histories[1] + env.action_histories[2]
        return torch.sum(diff ** 2 / env.step_dt ** 2, dim=1)
    
def joint_torques(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint torques on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.square(asset.data.applied_torque)
    return torch.sum(error, dim=1)

def joint_position_limit_penalty(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position limits on the articulation."""
    asset: Articulation = env.scene[asset_cfg.name]
    dof_pos_limits = asset.data.soft_joint_pos_limits
    dof_pos = asset.data.joint_pos

    out_of_limits = -(dof_pos - dof_pos_limits[..., 0]).clip(max=0.) # lower limit
    out_of_limits += (dof_pos - dof_pos_limits[..., 1]).clip(min=0.)

    return torch.sum(out_of_limits, dim=1)

def joint_velocity_penalty(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_vel), dim=1)

def joint_acceleration_penalty(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint accelerations on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_acc), dim=1)

def joint_position_penalty(
    env: HumanoidRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)

###############  Gait Specification Reward (Part1)  ###############

def air_time_reward(
    env: HumanoidRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    mode_time: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Reward longer feet air and contact time."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1).unsqueeze(dim=1).expand(-1, 2)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 2)
    reward = torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
        torch.where(t_max < mode_time, t_min, 0),
        stance_cmd_reward,
    )
    return torch.sum(reward, dim=1)

def foot_impact_penalty(env: HumanoidRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, cutoff: float) -> torch.Tensor:
    """Penalize foot impact when coming into contact with the ground"""
    asset: Articulation = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # get contact state
    is_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    # get velocity at contact
    foot_down_velocity = torch.clamp(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2], min=-abs(cutoff), max=0.0)
    # penalty is the velocity at contact squared when in contact
    reward = is_contact * torch.square(foot_down_velocity)
    return torch.sum(reward, dim=1)

def foot_slip_penalty(
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

def foot_clearance_reward(
    env: HumanoidRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    print("z target error: ", asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)

def air_time_variance_penalty(env: HumanoidRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )

def joint_regularization(env: HumanoidRLEnv) -> torch.Tensor:
    """Penalize joint symmetry. The joint indices are hard-coded for the T1 humanoid robot."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene["robot"]
    # regularize joint positions around default
    error = 0.
    # Yaw joints regularization around 0
    default_joint_pos = asset.data.default_joint_pos
    error += torch.square(
        (asset.data.joint_pos[:, 4]) - default_joint_pos[:, 4])
    error += torch.square(
        (asset.data.joint_pos[:, 5]) - default_joint_pos[:, 5])
    # Ab/ad joint symmetry
    error += torch.square(
        (asset.data.joint_pos[:, 2] - default_joint_pos[:, 2] + asset.data.joint_pos[:, 3] - default_joint_pos[:, 3]))
    # Pitch joint symmetry
    error += torch.square(
        (asset.data.joint_pos[:, 0] - default_joint_pos[:, 0] + asset.data.joint_pos[:, 1] - default_joint_pos[:, 1]))
    return error