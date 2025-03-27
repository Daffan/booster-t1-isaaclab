# IsaacLab Humanoid-Tasks
This IsaacLab extension implements a collections of Robocup Humanoid League tasks on a Booster T1 robot. List of implemented tasks:
- `Booster-T1-lb-walk-v0` (Velocity tracking custom reward function)
- `Booster-T1-lb-ROA-walk-v0` (Velocity tracking custom reward function with ROA)
- `Booster-T1-lb-ROA-walk-v1` (Velocity tracking [booster_gym](https://github.com/BoosterRobotics/booster_gym/blob/687a33d08b08875fe45dc8d91b54db83766df8b9/envs/t1.py#L606) reward function with ROA)
- `Booster-T1-lb-pre-kick-priv-v0` (Ball kicking pretraining with privileged info)

## Installation
This extension depends on NVIDIA's IsaacSim (v4.5.0) and IsaacLab (v2.0.0). Installation instructions can be found [here](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html). To install IsaacSim, the third step from this [instruction](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_python.html#installation-using-pip) is actually needed to avoid the error.

After installing the IsaacLab, install this extension using the following command:
```
pip install -e .
```

Setting up Visual Studio Code IDE is workspace is highly recommended, as it can give you type-hint for config files. More instructions can be found [here](https://isaac-sim.github.io/IsaacLab/main/source/overview/developer-guide/vs_code.html)

## Training script
```
./isaaclab.sh -p source/extensions/isaaclab.humanoid_tasks/scripts/train_roa.py --task Booster-T1-lb-ROA-walk-v1 --headless --video
```
The training by default is logged by `wandb`.

## Play script
```
./isaaclab.sh -p source/extensions/isaaclab.humanoid_tasks/scripts/play_roa.py --task Booster-T1-lb-ROA-walk-v1 --headless --video --load_run {YOUR/RUN/NAME}
```
Videos and related logs can be found under the path of loaded run.