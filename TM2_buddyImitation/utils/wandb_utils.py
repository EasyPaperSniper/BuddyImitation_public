# modified from rsl_rl

from __future__ import annotations

import os
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.")


class NoneSummaryWrite(SummaryWriter):
    def __init__(self,):
        pass

    def store_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        pass

    def _map_path(self, path):
        pass

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        pass

    def stop(self):
        pass

    def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        pass

    def save_model(self, model_path, iter):
        pass

    def save_file(self, path, iter=None):
        pass

class WandbSummaryWriter(SummaryWriter):
    """Summary writer for Weights and Biases."""

    def __init__(self, cfg, log_dir: str, flush_secs=15):
        # super().__init__(log_dir, flush_secs)

        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.")

        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            raise KeyError(
                "Wandb username not found. Please run or add to ~/.bashrc: export WANDB_USERNAME=YOUR_USERNAME"
            )


        wandb.init(project=cfg['wandb_project'], group=cfg['experiment_name'],name=cfg['run_name'],  entity=entity, dir=log_dir)

        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

        run_name = os.path.split(log_dir)[-1]

        wandb.log({"log_dir": run_name})

    def store_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        wandb.config.update({"runner_cfg": runner_cfg})
        wandb.config.update({"policy_cfg": policy_cfg})
        wandb.config.update({"alg_cfg": alg_cfg})
        wandb.config.update({"env_cfg": asdict(env_cfg)})


    def _map_path(self, path):
        if path in self.name_map:
            return self.name_map[path]
        else:
            return path

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        # super().add_scalar(
        #     tag,
        #     scalar_value,
        #     global_step=global_step,
        #     walltime=walltime,
        #     new_style=new_style,
        # )
        wandb.log({self._map_path(tag): scalar_value}, step=int(global_step))

    def stop(self):
        wandb.finish()

    def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        self.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)

    def save_model(self, model_path, iter):
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path, iter=None):
        wandb.save(path, base_path=os.path.dirname(path))
