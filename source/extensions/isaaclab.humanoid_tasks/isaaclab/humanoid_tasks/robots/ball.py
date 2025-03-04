import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

BALL_CFG = RigidObjectCfg(
        prim_path="/World/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.1,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.115),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.4, 0.0, 0.2),
        ),
    )