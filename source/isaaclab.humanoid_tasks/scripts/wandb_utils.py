#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import pathlib
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.")


class WandbSummaryWriter(SummaryWriter):
    """Summary writer for Weights and Biases."""

    def __init__(self, log_dir: str, flush_secs: int, cfg):
        super().__init__(log_dir, flush_secs)

        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.")

        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            raise KeyError(
                "Wandb username not found. Please run or add to ~/.bashrc: export WANDB_USERNAME=YOUR_USERNAME"
            )

        wandb.init(
            project=project,
            entity=entity
        )

        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

        run_name = os.path.split(log_dir)[-1]
        # Change generated name to project-number format
        wandb.run.name = log_dir.split("/")[1:].join("/")

        wandb.log({"log_dir": run_name})

        self.saved_videos = {}
        self.fps = 50

    def store_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        wandb.config.update({"runner_cfg": runner_cfg})
        wandb.config.update({"policy_cfg": policy_cfg})
        wandb.config.update({"alg_cfg": alg_cfg})
        wandb.config.update({"env_cfg": asdict(env_cfg)})

    def _map_path(self, path):
        if path in self.name_map:
            return self.name_map[path]
        else:
            return path

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        wandb.log({self._map_path(tag): scalar_value}, step=global_step)

    def log_video_files(self, log_name: str = "Video", video_subdir: str | None = "videos"):
        if video_subdir is not None:
            video_dir = pathlib.Path(os.path.join(self.log_dir, video_subdir))
        else:
            video_dir = pathlib.Path(self.log_dir)
        videos = list(video_dir.rglob("*.mp4"))
        for video in videos:
            video_name = str(video)
            video_size_kb = os.stat(video_name).st_size / 1024
            if video_name not in self.saved_videos.keys():
                self.saved_videos[video_name] = {"size": video_size_kb, "recorded": False, "steps": 0}
            else:
                video_info = self.saved_videos[video_name]
                if video_info["recorded"]:
                    continue
                elif video_info["size"] == video_size_kb and video_size_kb > 100:
                    # wait 10 steps after recording has been completed
                    if video_info["steps"] > 10:
                        self.add_video(video_name, fps=self.fps, log_name=log_name)
                        self.saved_videos[video_name]["recorded"] = True
                    else:
                        video_info["steps"] += 1
                else:
                    self.saved_videos[video_name]["size"] = video_size_kb
                    self.saved_videos[video_name]["steps"] = 0    

    def add_video(self, video_path: str, fps: int = 50, log_name: str = "Video"):
        wandb.log({log_name: wandb.Video(video_path, fps=fps)})    

    def callback(self, step):
        self.log_video_files()

    def stop(self):
        wandb.finish()

    def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        self.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)
        env_cfg_dict = asdict(env_cfg)
        self.fps = 1 / (env_cfg_dict["decimation"] * env_cfg_dict["sim"]["dt"])

    def save_model(self, model_path, iter):
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path, iter=None):
        wandb.save(path, base_path=os.path.dirname(path))

    def set_fps(self, fps):
        self.fps = fps
