import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject

def ball_reach_goal(env: ManagerBasedRLEnv,
                    ball_asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
                    goal_asset_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
                    goal_radius: float = 0.3) -> torch.Tensor:
    """Check if the ball is inside the goal"""
    ball_asset: RigidObject = env.scene[ball_asset_cfg.name]
    goal_asset: RigidObject = env.scene[goal_asset_cfg.name]
    goal_pos = goal_asset.data.root_pos_w[:, :2]
    ball_pos = ball_asset.data.root_pos_w[:, :2]
    return torch.norm(ball_pos - goal_pos, dim=-1) < goal_radius