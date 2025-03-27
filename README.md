# IsaacLab Training pipeline
This is the code base for debuging the failure of real-world deployment of the Booster T1 robot.

See `source/extensions/isaaclab.humanoid_tasks/README.md` for training script. 

A already trained model (JIT script) can be found under `logs/rsl_rl/t1-lb-ROA-walk/2025-03-23_16-31-50/exported/policy.pt`.

To play this model:
```
./isaaclab.sh -p source/extensions/isaaclab.humanoid_tasks/scripts/play_roa.py --task Booster-T1-lb-ROA-walk-v1 --headless --video --load_run 2025-03-23_16-31-50
```

Please refer to `booster_gym` branch for MuJoCo and real-world transfer script.