# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import numpy as np
import select
import sys
import mujoco, mujoco.viewer

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, EmpiricalNormalization

from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg


def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    import ipdb; ipdb.set_trace()

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # parse environment configuration
    # Hardcoded for now
    num_obs = 48
    num_critic_obs = 48
    num_actions = 12

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    actor_critic_class = eval(agent_cfg.policy.pop("class_name"))  # ActorCritic
    model: ActorCritic | ActorCriticRecurrent = actor_critic_class(
        num_obs, num_critic_obs, num_actions, **agent_cfg.policy
    ).to(agent_cfg.device)

    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    model_dict = torch.load(resume_path)
    model.load_state_dict(model_dict["model_state_dict"])

    # Set mujoco model
    # parse some parameters from config
    mujoco_file = "source/extensions/isaaclab.humanoid_tasks/isaaclab/humanoid_tasks/assets/mjcf/t1_locomotion.xml"
    default_joint_angles = env_cfg.scene.robot.init_state.joint_pos
    actuator_cfg = env_cfg.scene.robot.actuators
    joint_names = []
    stifnesses = []
    damping = []
    for k in actuator_cfg.keys():
        joint_names.extend(actuator_cfg[k].joint_names_expr)
        num_joints = len(actuator_cfg[k].joint_names_expr)
        stifnesses.extend([actuator_cfg[k].stiffness] * num_joints)
        damping.extend([actuator_cfg[k].damping] * num_joints)
    stiffness_dict = dict(zip(joint_names, stifnesses))
    damping_dict = dict(zip(joint_names, damping))

    mj_model = mujoco.MjModel.from_xml_path(mujoco_file)
    mj_model.opt.timestep = env_cfg["sim"]["dt"]
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    default_dof_pos = np.zeros(mj_model.nu, dtype=np.float32)
    dof_stiffness = np.zeros(mj_model.nu, dtype=np.float32)
    dof_damping = np.zeros(mj_model.nu, dtype=np.float32)
    for i in range(mj_model.nu):
        found = False
        for name in env_cfg.scene["default_joint_angles"].keys():
            if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                default_dof_pos[i] = default_joint_angles[name]

        for name in stiffness_dict.keys():
            if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                dof_stiffness[i] = stiffness_dict[name]
                dof_damping[i] = damping_dict[name]

    mj_data.qpos = np.concatenate(
        [
            np.array(env_cfg.scene.robot.init_state.pos, dtype=np.float32),
            np.array(env_cfg.scene.robot.init_state.rot[3:4] + env_cfg.scene.robot.init_state.rot[0:3], dtype=np.float32),
            default_dof_pos,
        ]
    )
    mujoco.mj_forward(mj_model, mj_data)

    actions = np.zeros(12, dtype=np.float32)
    dof_targets = np.zeros(default_dof_pos.shape, dtype=np.float32)
    gait_frequency = gait_process = 0.0
    lin_vel_x = lin_vel_y = ang_vel_yaw = 0.0
    it = 0

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.elevation = -20
        print(f"Set command (x, y, yaw): ")
        while viewer.is_running():
            if select.select([sys.stdin], [], [], 0)[0]:
                try:
                    parts = sys.stdin.readline().strip().split()
                    if len(parts) == 3:
                        lin_vel_x, lin_vel_y, ang_vel_yaw = map(float, parts)
                        if lin_vel_x == 0 and lin_vel_y == 0 and ang_vel_yaw == 0:
                            gait_frequency = 0
                        else:
                            gait_frequency = np.average(2.0)  # check the gait frequency here
                        print(
                            f"Updated command to: x={lin_vel_x}, y={lin_vel_y}, yaw={ang_vel_yaw}\nSet command (x, y, yaw): ",
                            end="",
                        )
                    else:
                        raise ValueError
                except ValueError:
                    print("Invalid input. Enter three numeric values.\nSet command (x, y, yaw): ", end="")
            dof_pos = mj_data.qpos.astype(np.float32)[7:]
            dof_vel = mj_data.qvel.astype(np.float32)[6:]
            quat = mj_data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float32)
            base_ang_vel = mj_data.sensor("angular-velocity").data.astype(np.float32)
            projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
            if it % env_cfg.decimation == 0:
                obs = np.zeros(num_obs, dtype=np.float32)
                obs[0:3] = projected_gravity # * cfg["normalization"]["gravity"]
                obs[3:6] = base_ang_vel # * cfg["normalization"]["ang_vel"]
                obs[6] = lin_vel_x # * cfg["normalization"]["lin_vel"]
                obs[7] = lin_vel_y # * cfg["normalization"]["lin_vel"]
                obs[8] = ang_vel_yaw # * cfg["normalization"]["ang_vel"]
                obs[9] = np.cos(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
                obs[10] = np.sin(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
                obs[11:23] = (dof_pos - default_dof_pos) # * cfg["normalization"]["dof_pos"]
                obs[23:35] = dof_vel * 0.1 # * cfg["normalization"]["dof_vel"]
                obs[35:47] = actions * 0.1
                dist = model.act(torch.tensor(obs).unsqueeze(0))
                actions[:] = dist.loc.detach().numpy()
                actions[:] = np.clip(actions, -1.0, 1.0)
                dof_targets[:] = default_dof_pos + actions
            mj_data.ctrl = np.clip(
                dof_stiffness * (dof_targets - dof_pos) - dof_damping * dof_vel,
                mj_model.actuator_ctrlrange[:, 0],
                mj_model.actuator_ctrlrange[:, 1],
            )
            mujoco.mj_step(mj_model, mj_data)
            viewer.cam.lookat[:] = mj_data.qpos.astype(np.float32)[0:3]
            viewer.sync()
            it += 1
            gait_process = np.fmod(gait_process + env_cfg.sim.dt * gait_frequency, 1.0)


if __name__ == "__main__":
    # run the main function
    main()