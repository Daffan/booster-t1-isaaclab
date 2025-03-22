# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

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
parser.add_argument("--viewer_scale", type=float, default=2.0, help="Viewer scale.")

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

from rsl_rl.runners import OnPolicyRunner

from isaaclab.humanoid_tasks.learning.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab.assets import Articulation, RigidObject

import isaaclab.humanoid_tasks.tasks
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, export_policy_as_jit, export_policy_as_onnx
from isaaclab.humanoid_tasks.learning.env.vecenv_wrapper import RslRlVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye=(12.5/args_cli.viewer_scale, 12.5/args_cli.viewer_scale, 7.5/args_cli.viewer_scale)
    env_cfg.events.randomize_joint_parameters = None
    env_cfg.scene.robot.spawn.articulation_props.fix_root_link = True  # fix root link to avoid falling

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # # wrap for video recording
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    test_result_dir = os.path.join("debug", "videos", timestamp)
    if args_cli.video:
        video_kwargs = {
            "video_folder": test_result_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    action_dim = env.action_space.shape[1]



    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    robot_asset: Articulation = env.unwrapped.scene["robot"]
    joint_names = robot_asset.data.joint_names
    T = int(args_cli.video_length / action_dim)

    # record the dof pos
    if True:
        dof_targets = []
        dof_limits = robot_asset.data.soft_joint_pos_limits.detach().cpu().numpy()[0, ...]
        dof_pos_list = []
        contacts = []
        obss = []
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            actions = torch.zeros((1, action_dim), device=env.device)
            joint_id = min(timestep // T, len(joint_names) - 1)
            if timestep % T == 0:
                print(f"Joint {joint_names[joint_id]}")
            actions[:, joint_id] = 0.5 * 4 * torch.sin(torch.tensor([timestep % T / T * 2 * np.pi]))

            # env stepping
            obs, rews, dones, infos = env.step(actions)
            # contacts.append(env.env.env.env.scene.sensors["contact_forces"].data.net_forces_w[0, [17, 23], 2].detach().cpu().numpy())
            dof_pos_list.append(obs[0, -36:-24].detach().cpu().numpy())
            dof_targets.append(robot_asset.data.joint_pos_target[0, :].detach().cpu().numpy())
            obss.append(obs[0, :].detach().cpu().numpy())
            # height
            # print(f"Height: {robot_asset.data.root_pos_w[0, 2].detach().cpu().numpy()}")
        timestep += 1
        if args_cli.video:
            # print(float(timestep) / args_cli.video_length)
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # plot dof pos
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 6, figsize=(32, 12))
    dof_targets = np.stack(dof_targets)
    # dof_pos = torch.stack(dof_pos, dim=1).detach().cpu().numpy().T
    dof_pos = np.stack(dof_pos_list)
    dim = min(25, action_dim)
    for i in range(dim):
        ax = axs[i//6, i%6]
        ax.plot(dof_targets[:, i], label='target')
        ax.plot(dof_pos[:, i], label='pos')
        # horizontal line for dof limits
        ax.axhline(dof_limits[i][0], color='r', linestyle='--')
        ax.axhline(dof_limits[i][1], color='r', linestyle='--')

        ax.set_title(f'DOF {i}: {joint_names[i]}')
        ax.legend()

    # save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(test_result_dir, 'dof.png'))
    plt.close()

    # plot all the observations
    fig, axs = plt.subplots(8, 8, figsize=(20, 20))
    obss = np.stack(obss)
    np.save(os.path.join(test_result_dir, 'obs.npy'), obss)
    dim = min(64, obss.shape[1])
    for i in range(dim):
        ax = axs[i//8, i%8]
        ax.plot(obss[:, i])
        ax.set_title(f'Obs {i}')

    # save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(test_result_dir, 'obs.png'))
    plt.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()