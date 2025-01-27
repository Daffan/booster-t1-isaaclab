import torch

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.assets import RigidObject

def humanoid_fall(env: ManagerBasedRLEnv,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                  max_pitch: float = 0.7,
                  max_roll: float = 0.7,
                  min_height: float = 0.3) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]
    projected_gravity = asset.data.projected_gravity_b
    base_pos = asset.data.root_pos_w
    return torch.any(torch.abs(projected_gravity[:, 0:1]) > max_pitch, dim=1) | \
        torch.any(torch.abs(projected_gravity[:, 1:2]) > max_roll, dim=1) | \
        torch.any(base_pos[:, 2:3] < min_height, dim=1)