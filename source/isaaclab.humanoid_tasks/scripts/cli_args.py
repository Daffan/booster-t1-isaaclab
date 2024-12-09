# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import yaml
import argparse
import random
import shutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", type=bool, default=None, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from omni.isaac.lab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg

def load_local_cfg(resume_path: str) -> dict:
    model_dir = os.path.dirname(resume_path)
    env_cfg_yaml_path = os.path.join(model_dir, "params", "env.yaml")
    # load yaml
    with open(env_cfg_yaml_path) as yaml_in:
        env_cfg = yaml.load(yaml_in, Loader=yaml.Loader)
    return env_cfg

def pull_policy_from_wandb(save_dir: str, run_path: str, model_name: str) -> tuple[str, dict]:
    import wandb

    # login to wandb
    wandb.login()
    api = wandb.Api()
    wandb_run = api.run(run_path)

    # download models to tmp_models folder
    wandb_folder_path = os.path.join(save_dir, "models")
    os.makedirs(wandb_folder_path, exist_ok=True)
    model_file = wandb_run.file(model_name)
    print(f"[INFO] Downloading model file to {wandb_folder_path}/{model_name}")
    model_file.download(f"{wandb_folder_path}", replace=True)
    resume_path = os.path.abspath(os.path.join(wandb_folder_path, model_name))
    # pull wandb model config
    print("[INFO] Pulling policy config from wandb")
    env_cfg = wandb_run.config["env_cfg"]
    return resume_path, env_cfg

def load_wandb_policy(run_path: str = "", model_name: str = "", log_root_path: str = "", log_dir: str = "") -> tuple[str, dict]:
    if run_path == "":
        run_path = input(
            "\033[96mEnter the Weights and Biases run path located on the Overview panel; i.e"
            " usr/Spot-Blind/abc123\033[0m\n"
        )
    while True:
        if model_name == "":
            model_name = input(
                "\n\033[96mEnter the name of the model file to download; i.e model_100.pt \n"
                + "Press Enter again without a file name to quit.\033[0m\n"
            )
        if model_name == "":
            return
        if model_name[:6] != "model_":
            model_name = "model_" + model_name
        if model_name[-3:] != ".pt":
            model_name += ".pt"
        try:
            resume_path, env_cfg = pull_policy_from_wandb(log_root_path, run_path, model_name)
            print(f"\033[92m\n[INFO] added policy to load\033[0m")
            model_file_name = os.path.splitext(os.path.basename(resume_path))[0]
            model_dir = os.path.join(log_dir, run_path.split("/")[-1])
            os.makedirs(model_dir, exist_ok=True)
            shutil.copy2(f"{resume_path}", f"{model_dir}/{model_file_name}.pt")
            break
        except Exception:
            print(
                "\n\033[91m[WARN] Unable to download from Weights and Biases, is the path"
                " and filename correct?\033[0m"
            )

    return resume_path, env_cfg