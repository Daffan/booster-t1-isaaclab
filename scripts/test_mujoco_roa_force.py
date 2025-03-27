import os
import sys
import glob
import yaml
import select
import argparse
import numpy as np
import torch
import mujoco, mujoco.viewer
from utils.model_roa import *
from collections import deque
import cv2


def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
    parser.add_argument("--checkpoint", type=str, help="Path of model checkpoint to load. Overrides config file if provided.")
    args = parser.parse_args()
    cfg_file = os.path.join("envs", "{}.yaml".format(args.task))
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    if args.checkpoint is not None:
        cfg["basic"]["checkpoint"] = args.checkpoint

    if not cfg["basic"]["checkpoint"] or (cfg["basic"]["checkpoint"] == "-1") or (cfg["basic"]["checkpoint"] == -1):
        cfg["basic"]["checkpoint"] = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
    print("Loading model from {}".format(cfg["basic"]["checkpoint"]))
    
    ac_model = torch.jit.load(cfg["basic"]["checkpoint"], map_location="cpu")
    policy = lambda x : ac_model.act_mj(x)

    mj_model = mujoco.MjModel.from_xml_path(cfg["asset"]["mujoco_file"])
    mj_model.opt.timestep = cfg["sim"]["dt"]
    mj_data = mujoco.MjData(mj_model)

    renderer = mujoco.Renderer(mj_model, width=640, height=480)
    scene = mujoco.MjvScene(mj_model, maxgeom=1000)
    camera = mujoco.MjvCamera()  # Camera object
    mujoco.mjv_defaultCamera(camera)
    # Create an `mjvOption` object to pass into `mjv_updateScene`
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)  # Initialize options

    mujoco.mj_resetData(mj_model, mj_data)
    default_dof_pos = np.zeros(mj_model.nu, dtype=np.float32)
    dof_stiffness = np.zeros(mj_model.nu, dtype=np.float32)
    dof_damping = np.zeros(mj_model.nu, dtype=np.float32)
    for i in range(mj_model.nu):
        found = False
        for name in cfg["init_state"]["default_joint_angles"].keys():
            if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                print(name)
                default_dof_pos[i] = cfg["init_state"]["default_joint_angles"][name]
                found = True
        if not found:
            default_dof_pos[i] = cfg["init_state"]["default_joint_angles"]["default"]

        found = False
        for name in cfg["control"]["stiffness"].keys():
            if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                dof_stiffness[i] = cfg["control"]["stiffness"][name]
                dof_damping[i] = cfg["control"]["damping"][name]
                found = True
        if not found:
            raise ValueError(f"PD gain of joint {mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)} were not defined")
    mj_data.qpos = np.concatenate(
        [
            np.array(cfg["init_state"]["pos"], dtype=np.float32),
            np.array(cfg["init_state"]["rot"][3:4] + cfg["init_state"]["rot"][0:3], dtype=np.float32),
            default_dof_pos,
        ]
    )
    mujoco.mj_forward(mj_model, mj_data)

    actions = np.zeros((cfg["env"]["num_actions"]), dtype=np.float32)
    dof_targets = np.zeros(default_dof_pos.shape, dtype=np.float32)
    gait_frequency = gait_process = 1.0
    lin_vel_x = 0.5
    lin_vel_y = ang_vel_yaw = 0.0
    it = 0
    history = 50
    obss = deque(maxlen=history)

    frame_width = 640
    frame_height = 480
    video_writer = cv2.VideoWriter(
        "debug/test_force.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 50, (frame_width, frame_height)
    )

    force_schedule = np.array([
        [1.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [20.0, 20.0, 0.0],
        [50.0, 10.0, 0.0],
        [100.0, 0.0, 0.0],
    ])

    body_name = "Trunk"  # Change this to the name of the base body
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    force_magnitude = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mj_data.xfrc_applied[body_id] = force_magnitude

    # with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    #     viewer.cam.elevation = -20
    #     while viewer.is_running():
    while True:
            if it % 2000 == 0:
                if it // 2000 >= len(force_schedule):
                    break
                else:
                    fx, fy, fz = force_schedule[it // 2000]
                    force_magnitude[0] = fx; force_magnitude[1] = fy; force_magnitude[2] = fz
                    print(f"Updated force to: x={fx}, y={fy}, z={fz}")
                    
            dof_pos = mj_data.qpos.astype(np.float32)[7:]
            dof_vel = mj_data.qvel.astype(np.float32)[6:]
            quat = mj_data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float32)
            base_ang_vel = mj_data.sensor("angular-velocity").data.astype(np.float32)
            projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))

            if it % cfg["control"]["decimation"] == 0:
                obs = np.zeros(cfg["env"]["num_prop"], dtype=np.float32)
                obs[0:3] = projected_gravity * cfg["normalization"]["gravity"]
                obs[3:6] = base_ang_vel * cfg["normalization"]["ang_vel"]
                obs[6] = lin_vel_x * cfg["normalization"]["lin_vel"]
                obs[7] = lin_vel_y * cfg["normalization"]["lin_vel"]
                obs[8] = ang_vel_yaw * cfg["normalization"]["ang_vel"]
                # obs[9] = gait_frequency
                obs[10-1] = np.cos(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
                obs[11-1] = np.sin(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
                obs[12-1:24-1] = (dof_pos - default_dof_pos) * cfg["normalization"]["dof_pos"]
                obs[24-1:36-1] = dof_vel * cfg["normalization"]["dof_vel"]
                obs[36-1:48-1] = actions
                if len(obss) == 0:
                    obss.extend([np.zeros_like(obs)] * history) 
                obss.append(obs)  # append the current frame
                out = policy(torch.tensor(np.stack(obss)).unsqueeze(0)).squeeze(0)
                actions[:] = out.detach().numpy()
                actions[[5, 11]] = 0.0
                actions[:] = np.clip(actions, -cfg["normalization"]["clip_actions"], cfg["normalization"]["clip_actions"])
                dof_targets[:] = default_dof_pos + cfg["control"]["action_scale"] * actions

            mj_data.ctrl = np.clip(
                dof_stiffness * (dof_targets - dof_pos) - dof_damping * dof_vel,
                mj_model.actuator_ctrlrange[:, 0],
                mj_model.actuator_ctrlrange[:, 1],
            )
            mj_data.xfrc_applied[body_id] = force_magnitude
            mujoco.mj_step(mj_model, mj_data)
            # viewer.cam.lookat[:] = mj_data.qpos.astype(np.float32)[0:3]
            # viewer.sync()

            camera.lookat[:] = mj_data.qpos.astype(np.float32)[0:3]
            camera.distance = 2.0  # Adjust viewing distance
            camera.azimuth = 90  # Set viewing angle
            camera.elevation = -20  # Set elevation angle
            
            it += 1
            gait_process = np.fmod(gait_process + cfg["sim"]["dt"] * gait_frequency, 1.0)

            if it % 10 == 0:
                # frame = np.flipud(frame)
                # add text for current command
                mujoco.mjv_updateScene(
                    mj_model,  # Model
                    mj_data,   # Data
                    opt,       # Visualization options
                    None,      # No perturbation
                    camera,    # Camera object
                    mujoco.mjtCatBit.mjCAT_ALL,  # Render all objects
                    scene      # Scene object
                )

                renderer.update_scene(mj_data, camera)  # Now update the renderer with the scene
                frame = renderer.render()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    frame_bgr,
                    f"Force: x={force_magnitude[0]}, y={force_magnitude[1]}, z={force_magnitude[2]}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)  # Green text
                video_writer.write(frame_bgr)

    video_writer.release()
    print("Video saved to 'debug/test_force.mp4'")
    print("Done.")