from __future__ import annotations

from typing import TYPE_CHECKING
from collections import OrderedDict as odict

from copy import deepcopy

from numpy import indices
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import resolve_matching_names_values

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import matplotlib.pyplot as plt

class Plotter:
    def __init__(
            self,
            env: ManagerBasedRLEnv,
            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*"], preserve_order=True),
            sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=[".*_foot"]),
            store_estimate: bool = False,
            store_dynamics: bool = False
    ) -> None:
        self.env = env
        asset: Articulation = env.scene[asset_cfg.name]
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        self.store_estimate = store_estimate
        self.store_dynamics = store_dynamics
        # Data
        self.Ka = env.cfg.actions.joint_pos.scale
        # Time
        self.dt = env.cfg.sim.dt * env.cfg.decimation
        self.t = []
        self.counter = 0
        # q_stance
        q_stance_re = asset.cfg.init_state.joint_pos
        # ret = (indices, names, values)
        ret = resolve_matching_names_values(q_stance_re, asset.joint_names, preserve_order=True)
        q_stance_list = [None] * len(ret[0])
        for idx, k, v in zip(*ret):
            q_stance_list[idx] = (k, v)
        self.q_stance = odict(q_stance_list)

        empty_dict = odict(
            [(k, []) for k in asset.joint_names]
        )
        # Latents Space
        self.latents = {
            i: [] for i in range(8)
        }
        # Velocity
        self.vel = {
            "vx": [],
            "vy": [],
            "vz": [],
            "avx": [],
            "avy": [],
            "avz": [],
        }
        # Projected Gravity
        self.proj_g = {
            "gx": [],
            "gy": [],
            "gz": [],
        }
        # Velocity Command
        self.vel_cmd = {
            "vx": [],
            "vy": [],
            "yaw rate": [],
        }
        # Joint Position
        self.q = deepcopy(empty_dict)
        self.q_des = deepcopy(empty_dict)
        # Joint Velocity
        self.dq = deepcopy(empty_dict)
        # Action
        self.action = deepcopy(empty_dict)
        self.torque = deepcopy(empty_dict)
        # Estimate
        if self.store_estimate:
            self.estimate = deepcopy(empty_dict)
        # Wrench
        if self.store_dynamics:
            self.wrench = {
                "Fx": [],
                "Fy": [],
                "Fz": [],
                "Tx": [],
                "Ty": [],
                "Tz": [],
            }
            self.grf = odict()
            _, grf_bodies = sensor.find_bodies(sensor_cfg.body_names, preserve_order=True)
            for k in grf_bodies:
                k = k.split("_")[0]
                self.grf[k+"x"] = []
                self.grf[k+"y"] = []
                self.grf[k+"z"] = []

        # Plotters
        subplot_kw_args = {
            "sharex": True,
            # "sharey": True,
        }
        # Latents
        self.fig_latents, self.ax_latents = plt.subplots(1, 1, **subplot_kw_args)
        # Velocity
        self.fig_vel, self.ax_vel = plt.subplots(2, 1, **subplot_kw_args)
        # Projected Gravity
        self.fig_proj_g, self.ax_proj_g = plt.subplots()
        # Velocity Command
        self.fig_vel_cmd, self.ax_vel_cmd = plt.subplots()
        # Joint Position
        self.fig_q, self.ax_q = plt.subplots(3, 1, **subplot_kw_args)
        # Joint Velocity
        self.fig_dq, self.ax_dq = plt.subplots(3, 1, **subplot_kw_args)
        # Action
        self.fig_action, self.ax_action = plt.subplots(3, 1, **subplot_kw_args)
        # Torque
        self.fig_torque, self.ax_torque = plt.subplots(3, 1, **subplot_kw_args)
        # Estimate
        if self.store_estimate:
            self.fig_est, self.ax_est = plt.subplots(1, 1, **subplot_kw_args)
        # Wrench
        if store_dynamics:
            self.fig_wrench, self.ax_wrench = plt.subplots(2, 1, **subplot_kw_args)
            self.fig_grf, self.ax_grf = plt.subplots(3, 1, **subplot_kw_args)


    def log(
            self,
            lin_vel: list,
            ang_vel: list,
            proj_g: list,
            vel_cmd: list,
            q: list,
            dq: list,
            action: list,
            torque: list,
            q_des: list,
            estimates: list = None,
            wrenches: list = None,
            GRFs: list = None,
            latents: list = None,
    ) -> None:
        # Time
        self.t.append(self.counter * self.dt)
        self.counter += 1
        # Linear Velocity
        self.vel["vx"].append(lin_vel[0])
        self.vel["vy"].append(lin_vel[1])
        self.vel["vz"].append(lin_vel[2])
        # Angular Velocity
        self.vel["avx"].append(ang_vel[0])
        self.vel["avy"].append(ang_vel[1])
        self.vel["avz"].append(ang_vel[2])
        # Projected Gravity
        self.proj_g["gx"].append(proj_g[0])
        self.proj_g["gy"].append(proj_g[1])
        self.proj_g["gz"].append(proj_g[2])
        # Velocity Command
        self.vel_cmd["vx"].append(vel_cmd[0])
        self.vel_cmd["vy"].append(vel_cmd[1])
        self.vel_cmd["yaw rate"].append(vel_cmd[2])
        for idx, k in enumerate(self.q.keys()):
            # Joint Position
            self.q[k].append(q[idx] + self.q_stance[k])
            if q_des is not None:
                self.q_des[k].append(q_des[idx])
            else:
                self.q_des[k].append(self.Ka * action[idx] + self.q_stance[k])
            # Joint Velocity
            self.dq[k].append(dq[idx])
            # Action
            self.action[k].append(self.Ka * action[idx])
            # Torque
            self.torque[k].append(torque[idx])
            # Estimate
            if self.store_estimate and estimates is not None:
                # TODO: add logic to handle any estimate
                self.estimate[k].append(estimates[idx])
                # TODO: add ground truth values for estimates
        if latents is not None:
            for i in range(len(latents)):
                self.latents[i].append(latents[i])
        if self.store_dynamics:
            # COM Wrenches
            if wrenches is not None:
                self.wrench["Fx"].append(wrenches[0])
                self.wrench["Fy"].append(wrenches[1])
                self.wrench["Fz"].append(wrenches[2])
                self.wrench["Tx"].append(wrenches[3])
                self.wrench["Ty"].append(wrenches[4])
                self.wrench["Tz"].append(wrenches[5])
            # Foot GRFs
            if GRFs is not None:
                for idx, k in enumerate(self.grf.keys()):
                    self.grf[k].append(GRFs[idx])
    
    def plot(self, path: str) -> None:
        if path[-1] != "/": path += "/"
        # set removes duplicates
        style_keys = list({k.split("_")[0] for k in self.q.keys() if len(k.split("_")) > 1})
        style_values = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        styles = {
            k: v for k, v in zip(style_keys, style_values)
        }
        # Velocity
        for k, v in self.vel.items():
            if "a" in k:
                self.ax_vel[1].plot(self.t, v, label=k)
            else:
                self.ax_vel[0].plot(self.t, v, label=k)
        self.ax_vel[0].legend(loc="upper right")
        self.ax_vel[1].legend(loc="upper right")
        self.ax_vel[0].set_title("Velocity")
        self.ax_vel[0].set_ylabel("Linear Velocity [m/s]")
        self.ax_vel[1].set_ylabel("Angular Velocity [rad/s]")
        self.ax_vel[1].set_xlabel("Time [s]")
        self.fig_vel.savefig(path + "velocity.png")
        # Projected Gravity
        for k, v in self.proj_g.items():
            self.ax_proj_g.plot(self.t, v, label=k)
        self.ax_proj_g.legend(loc="upper right")
        self.ax_proj_g.set_title("Projected Gravity")
        self.ax_proj_g.set_ylabel("Force [N]")
        self.ax_proj_g.set_xlabel("Time [s]")
        self.fig_proj_g.savefig(path + "proj_g.png")
        # Velocity Command
        for k, v in self.vel_cmd.items():
            self.ax_vel_cmd.plot(self.t, v, label=k)
        self.ax_vel_cmd.legend(loc="upper right")
        self.ax_vel_cmd.set_title("Velocity Command")
        self.ax_vel_cmd.set_ylabel("Velocity [m/s, rad/s]")
        self.ax_vel_cmd.set_xlabel("Time [s]")
        self.fig_vel_cmd.savefig(path + "vel_cmd.png")
        # Joint Position
        for k, v in self.q.items():
            c = styles[k.split("_")[0]]
            if "hip" in k or "hx" in k:
                self.ax_q[0].plot(self.t, v, c, label=k.split("_")[0].upper())
                self.ax_q[0].legend(loc="upper right")
            elif "thigh" in k or "hy" in k:
                self.ax_q[1].plot(self.t, v, c, label=k.split("_")[0].upper())
                self.ax_q[1].legend(loc="upper right")
            elif "calf" in k or "kn" in k:
                self.ax_q[2].plot(self.t, v, c, label=k.split("_")[0].upper())
                self.ax_q[2].legend(loc="upper right")
        # Desired
        for k, v in self.q_des.items():
            c = styles[k.split("_")[0]]
            if "hip" in k or "hx" in k:
                self.ax_q[0].plot(self.t, v, c, ls="--")
                self.ax_q[0].legend(loc="upper right")
            elif "thigh" in k or "hy" in k:
                self.ax_q[1].plot(self.t, v, c, ls="--")
                self.ax_q[1].legend(loc="upper right")
            elif "calf" in k or "kn" in k:
                self.ax_q[2].plot(self.t, v, c, ls="--")
                self.ax_q[2].legend(loc="upper right")
        self.ax_q[0].set_title("Joint Position")
        self.ax_q[1].set_ylabel("Angle [rad]")
        self.ax_q[2].set_xlabel("Time [s]")
        self.fig_q.savefig(path + "q.png")
        # Joint Velocity
        for k, v in self.dq.items():
            c = styles[k.split("_")[0]]
            if "hip" in k or "hx" in k:
                self.ax_dq[0].plot(self.t, v, c, label=k.split("_")[0].upper())
                self.ax_dq[0].legend(loc="upper right")
            elif "thigh" in k or "hy" in k:
                self.ax_dq[1].plot(self.t, v, c, label=k.split("_")[0].upper())
                self.ax_dq[1].legend(loc="upper right")
            elif "calf" in k or "kn" in k:
                self.ax_dq[2].plot(self.t, v, c, label=k.split("_")[0].upper())
                self.ax_dq[2].legend(loc="upper right")
        self.ax_dq[0].set_title("Joint Velocity")
        self.ax_dq[1].set_ylabel("Angular Rate [rad/s]")
        self.ax_dq[2].set_xlabel("Time [s]")
        self.fig_dq.savefig(path + "dq.png")
        # Action
        for k, v in self.action.items():
            c = styles[k.split("_")[0]]
            if "hip" in k or "hx" in k:
                self.ax_action[0].plot(self.t, v, c, label=k.split("_")[0].upper())
                self.ax_action[0].legend(loc="upper right")
            elif "thigh" in k or "hy" in k:
                self.ax_action[1].plot(self.t, v, c, label=k.split("_")[0].upper())
                self.ax_action[1].legend(loc="upper right")
            elif "calf" in k or "kn" in k:
                self.ax_action[2].plot(self.t, v, c, label=k.split("_")[0].upper())
                self.ax_action[2].legend(loc="upper right")
        self.ax_action[0].set_title("Action")
        self.ax_action[1].set_ylabel("Angle Offset [rad]")
        self.ax_action[2].set_xlabel("Time [s]")
        self.fig_action.savefig(path + "action.png")
        # Torque
        for k, v in self.torque.items():
            c = styles[k.split("_")[0]]
            if "hip" in k or "hx" in k:
                self.ax_torque[0].plot(self.t, v, c, label=k.split("_")[0].upper())
                self.ax_torque[0].legend(loc="upper right")
            elif "thigh" in k or "hy" in k:
                self.ax_torque[1].plot(self.t, v, c, label=k.split("_")[0].upper())
                self.ax_torque[1].legend(loc="upper right")
            elif "calf" in k or "kn" in k:
                self.ax_torque[2].plot(self.t, v, c, label=k.split("_")[0].upper())
                self.ax_torque[2].legend(loc="upper right")
        self.ax_torque[0].set_title("Applied Torque")
        self.ax_torque[1].set_ylabel("Torque [Nm]")
        self.ax_torque[2].set_xlabel("Time [s]")
        self.fig_torque.savefig(path + "torque.png")
        # Estimates
        if self.store_estimate:
            for k, v in self.estimate.items():
                c = styles[k.split("_")[0]]
                self.ax_est[0].plot(self.t, v, c, label=k)
                self.ax_est[0].legend(loc="upper right")
            self.ax_est[0].set_title("Estimates")
            self.ax_est[0].set_ylabel("Estimate")
            self.ax_est[0].set_xlabel("Time [s]")
            self.fig_est.savefig(path + "estimate.png")
        if self.store_dynamics:
            # COM Wrenches
            for k, v in self.wrench.items():
                if "F" in k:
                    self.ax_wrench[0].plot(self.t, v, label=k)
                elif "T" in k:
                    self.ax_wrench[1].plot(self.t, v, label=k)
            self.ax_wrench[0].legend(loc="upper right")
            self.ax_wrench[1].legend(loc="upper right")
            self.ax_wrench[0].set_title("External Wrench")
            self.ax_wrench[0].set_ylabel("Force [N]")
            self.ax_wrench[1].set_ylabel("Torque [Nm]")
            self.ax_wrench[1].set_xlabel("Time [s]")
            self.fig_wrench.savefig(path + "wrench.png")
            # Foot GRFs
            for k, v in self.grf.items():
                for idx, ax in enumerate(["x", "y", "z"]):
                    if ax in k:
                        self.ax_grf[idx].plot(self.t, v, label=k.split(ax)[0].upper())
            self.ax_grf[0].legend(loc="upper right")
            self.ax_grf[1].legend(loc="upper right")
            self.ax_grf[2].legend(loc="upper right")
            self.ax_grf[0].set_title("Ground Reaction Force")
            self.ax_grf[0].set_ylabel("Fx [N]")
            self.ax_grf[1].set_ylabel("Fy [N]")
            self.ax_grf[2].set_ylabel("Fz [N]")
            self.ax_grf[2].set_xlabel("Time [s]")
            self.fig_grf.savefig(path + "grf.png")
        for k, v in self.latents.items():
            self.ax_latents.plot(self.t, v, label=k)
        self.fig_latents.savefig(path + "latents.png")