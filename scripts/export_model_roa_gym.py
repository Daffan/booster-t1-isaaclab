import os
import glob
import yaml
import argparse
import torch
from utils.model_roa_gym import *

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

    model = ActorCriticHistory(
        cfg["env"]["num_observations"],
        cfg["env"]["num_privileged_obs"],
        cfg["env"]["num_actions"],
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 256, 128],
        priv_encoder_dims=[64, 20],
        num_priv=cfg["env"]["num_priv"],
        num_hist=cfg["env"]["hist_len"],
        num_prop=cfg["env"]["num_prop"]
    )
    if not cfg["basic"]["checkpoint"] or (cfg["basic"]["checkpoint"] == "-1") or (cfg["basic"]["checkpoint"] == -1):
        cfg["basic"]["checkpoint"] = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
    print("Loading model from {}".format(cfg["basic"]["checkpoint"]))
    model_dict = torch.load(cfg["basic"]["checkpoint"], map_location="cpu", weights_only=True)
    model_dict = model_dict["model"]
    model_dict_new = {}
    for key in list(model_dict.keys()):
        if key.startswith("actor."):
            # model_dict_new[key.replace("actor.", "")] = model_dict[key]
            model_dict_new[key[6:]] = model_dict[key]
    model.actor.load_state_dict(model_dict_new, strict=False)

    model.eval()
    script_module = torch.jit.script(model.actor)
    save_path = os.path.splitext(cfg["basic"]["checkpoint"])[0] + "_export" + ".pt"
    script_module.save(save_path)
    print(f"Saved model to {save_path}")
