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

from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

import isaaclab.humanoid_tasks.tasks


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye=(12.5/3, 12.5/3, 7.5/3)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # # wrap for video recording
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    test_result_dir = os.path.join(log_dir, "videos", timestamp)
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

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    # record the dof pos
    if True:
        dof_targets = []
        dof_limits = env.env.env.env.scene["robot"].data.soft_joint_pos_limits.detach().cpu().numpy()[0, ...]
        contacts = []
        obss = []

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            
            # debug
            # T = 80
            # actions = torch.zeros_like(actions)
            # if timestep // T == 0:
            #     actions[:, 0] = 0.5 * 4 * torch.sin(torch.tensor([timestep % T / T * 2 * np.pi]))
            #     actions[:, 1] = -0.5 * 4 * torch.sin(torch.tensor([timestep % T / T * 2 * np.pi]))
            # elif timestep // T == 1:
            #     actions[:, 2] = 0.5 * 4 * torch.sin(torch.tensor([timestep % T / T * 2 * np.pi]))
            #     actions[:, 3] = -0.5 * 4 * torch.sin(torch.tensor([timestep % T / T * 2 * np.pi]))
            # elif timestep // T == 2:
            #     actions[:, 4] = 0.5 * 4 * torch.sin(torch.tensor([timestep % T / T * 2 * np.pi]))
            #     actions[:, 5] = -0.5 * 4 * torch.sin(torch.tensor([timestep % T / T * 2 * np.pi]))
            # elif timestep // T == 3:
            #     actions[:, 6] = 0.5 * 4 * torch.sin(torch.tensor([timestep % T / T * 2 * np.pi]))
            #     actions[:, 7] = -0.5 * 4 * torch.sin(torch.tensor([timestep % T / T * 2 * np.pi]))
            # elif timestep // T == 4:
            #     actions[:, 8] = 0.5 * 4 * torch.sin(torch.tensor([timestep % T / T * 2 * np.pi]))
            #     actions[:, 9] = -0.5 * 4 * torch.sin(torch.tensor([timestep % T / T * 2 * np.pi]))
            dof_targets.append(actions[0, :].detach().cpu().numpy())

            # if play isaacgym policy
            # actions = actions[:, [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]]

            # env stepping
            obs, _, _, _ = env.step(actions)
            contacts.append(env.env.env.env.scene.sensors["contact_forces"].data.net_forces_w[0, [17, 23], 2].detach().cpu().numpy())
            obss.append(obs[0, :].detach().cpu().numpy())
        if args_cli.video:
            # print(float(timestep) / args_cli.video_length)
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # plot dof pos
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 4, figsize=(20, 12))
    dof_targets = np.stack(dof_targets)
    # dof_pos = torch.stack(dof_pos, dim=1).detach().cpu().numpy().T
    dof_pos = np.stack(obss)[:, 13:23]
    for i in range(10):
        ax = axs[i//4, i%4]
        ax.plot(dof_targets[:, i], label='target')
        ax.plot(dof_pos[:, i], label='pos')
        # horizontal line for dof limits
        ax.axhline(dof_limits[i][0], color='r', linestyle='--')
        ax.axhline(dof_limits[i][1], color='r', linestyle='--')

        ax.set_title(f'DOF {i}')
        ax.legend()

    # save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(test_result_dir, 'dof.png'))
    plt.close()

    fig, axs = plt.subplots(3, 4, figsize=(20, 12))
    for i in range(5):
        ax = axs[i//4, i%4]
        ax.plot(dof_targets[:, i*2], label='target')
        ax.plot(dof_pos[:, i*2], label='pos')
        # horizontal line for dof limits
        ax.axhline(dof_limits[i*2][0], color='r', linestyle='--')
        ax.axhline(dof_limits[i*2][1], color='r', linestyle='--')

        ax.plot(dof_targets[:, i*2+1], label='target')
        ax.plot(dof_pos[:, i*2+1], label='pos')

        ax.set_title(f'DOF {i}')
        ax.legend()

    # plot contact
    contacts = np.stack(contacts)
    obss = np.stack(obss)
    ax1 = axs[2, 3]
    ax1.plot(contacts[:, 0], label='left')
    ax2 = axs[2, 2]
    ax2.plot(contacts[:, 1], label='right')
    ax2.set_title('Contact')

    ax3 = axs[2, 1]
    ax3.plot(obss[:, -2], label='sin')
    ax3.set_title('phase')

    ax4 = axs[2, 0]
    ax4.plot(obss[:, -1], label='cos')
    ax4.set_title('phase')

    # save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(video_kwargs["video_folder"], 'dof_compare.png'))
    plt.close()

    # plot obs histories [38]

    # load reference obs data
    if os.path.exists(os.path.join(video_kwargs["video_folder"], "../../..", 'play_log.csv')):
        reference_data = np.loadtxt(os.path.join(video_kwargs["video_folder"], "../../..", 'play_log.csv'), delimiter=",")
        reference_obs = reference_data[:, 1:39]
    else:
        reference_obs = None

    fig, axs = plt.subplots(6, 7, figsize=(20, 12))
    for i in range(38):
        ax = axs[i//7, i%7]
        ax.plot(obss[:, i])
        if reference_obs is not None:
            ax.plot(reference_obs[:, i], color='r')
        ax.set_title(f'Obs {i}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(video_kwargs["video_folder"], 'obs.png'))
    plt.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
