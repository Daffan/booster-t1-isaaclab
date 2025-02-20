# IsaacLab Humanoid-Tasks
This IsaacLab extension implements a collections of Robocup Humanoid League tasks on a Booster T1 robot. Full list of implemented tasks:
- [x] `RL-Booster-t1-walking-v0` (Velocity tracking)
- [x] `RL-Booster-t1-kick-v0` (Ball kicking)
- [ ] Getting up from ground
- [ ] Head Positioning

## Installation
This extension depends on NVIDIA's IsaacSim (v4.2.0) and IsaacLab (v1.3.0). Installation instructions can be found [here](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html). To install IsaacSim, the third step from this [instruction](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_python.html#installation-using-pip) is actually needed to avoid the error. 
**Note that this repo is not compatible with newer version of IsaacLab (>= v2.0).**

After installing the IsaacLab, install this extension using the following command:
```
pip install -e .
```

Setting up Visual Studio Code IDE is workspace is highly recommended, as it can give you type-hint for config files. More instructions can be found [here](https://isaac-sim.github.io/IsaacLab/main/source/overview/developer-guide/vs_code.html)

## Training script
```
./isaaclab.sh -p source/extensions/isaaclab.humanoid_tasks/scripts/train.py --task RL-Booster-t1-walking-v0 --headless --video
```
The training by default is logged by `wandb`.

## Play script
```
./isaaclab.sh -p source/extensions/isaaclab.humanoid_tasks/scripts/play.py --task RL-Booster-t1-walking-v0 --headless --video --load_run {YOUR/RUN/NAME}
```
Videos and related logs can be found under the path of loaded run.

## TODO List
- [ ] Sim-to-sim transfer pipeline.
- [ ] Sim-to-real transfer pipeline.
- [ ] Adjust the sizes of primitive shapes in `source/isaaclab.humanoid_tasks/isaaclab/humanoid_tasks/assets/urdf/T1_Sim_fixed_arms.urdf` (current size is not accurate and may cause direct collision if self-collision is turned on).
- [ ] Create robot asset with controllable upper-body joints.

## Robots
We provide the following robot actuators:

* Booster T1 with fixed upper-body joints:

`source/extensions/isaaclab.humanoid_tasks/isaaclab/humanoid_tasks/robots/t1.py`

| Joint Name        | Index |
|-------------------|-------|
| Left_Hip_Pitch    | 0     |
| Right_Hip_Pitch   | 1     |
| Left_Hip_Roll     | 2     |
| Right_Hip_Roll    | 3     |
| Left_Hip_Yaw      | 4     |
| Right_Hip_Yaw     | 5     |
| Left_Knee_Pitch   | 6     |
| Right_Knee_Pitch  | 7     |
| Left_Ankle_Pitch  | 8     |
| Right_Ankle_Pitch | 9     |


* Booster T1 whole-body control:

`source/extensions/isaaclab.humanoid_tasks/isaaclab/humanoid_tasks/robots/t1_wb.py`

## Trouble Shooting
