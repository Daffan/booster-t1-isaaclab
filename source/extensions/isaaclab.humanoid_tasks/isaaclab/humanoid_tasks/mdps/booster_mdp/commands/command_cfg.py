from collections.abc import Sequence
from typing import Any
from dataclasses import MISSING

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs.mdp import UniformVelocityCommandCfg
# from omni.isaac.lab.envs.mdp import UniformPose2dCommand, UniformPose2dCommandCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
import omni.isaac.lab.sim as sim_utils

from isaaclab.humanoid_tasks.envs import HumanoidRLEnv

# from .command_cfg import ReplayPose2dCommandCfg

@configclass
class UniformVelocityFreqCommandCfg(UniformVelocityCommandCfg):
    filter_weight: float = 0.1

    @configclass
    class Ranges(UniformVelocityCommandCfg.Ranges):
        """Uniform distribution ranges for the velocity commands."""

        gait_frequency: tuple[float, float] = MISSING
        """Periodic gait frequency range in Hz. Suggest from [1, 2] Hz."""

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""