from collections.abc import Sequence
from typing import Any
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.envs.mdp import UniformVelocityCommandCfg
# from isaaclab.envs.mdp import UniformPose2dCommand, UniformPose2dCommandCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
import isaaclab.sim as sim_utils

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