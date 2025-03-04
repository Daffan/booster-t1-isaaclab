from dataclasses import MISSING

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

@configclass
class ClippedJointPositionActionCfg(mdp.JointPositionActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = MISSING

    use_default_offset: bool = True
    """Whether to use default joint positions configured in the articulation asset as offset.
    Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to the default joint positions
    from the articulation asset.
    """

    action_clip: float = 1.0


class ClippedJointPositionAction(mdp.JointPositionAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: ClippedJointPositionActionCfg
    """The configuration of the action term."""

    def apply_actions(self):
        # apply the processed actions
        # print("actions", self._processed_actions, self._processed_actions.clamp(-self.cfg.clip, self.cfg.clip))
        self._asset.set_joint_position_target(
            self._processed_actions.clamp(-self.cfg.action_clip, self.cfg.action_clip),
            joint_ids=self._joint_ids
        )