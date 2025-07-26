import os
import copy
import random
from datetime import datetime
import argparse

import torch
import numpy as np


from TM2_buddyImitation import TM2_RESULT_DIR, TM2_ROOT_DIR, TM2_ENVS_DIR




def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result



def get_args():
    parser = argparse.ArgumentParser(description="TM2_buddyImitation")

    # env setting
    env_group = parser.add_argument_group("Env", description="Arguments for Env Setting.")
    env_group.add_argument("--device", type=str, default=None, help="Use CPU pipeline.")
    env_group.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    env_group.add_argument("--render", type=bool, default=False, help="Render the environment.")
    env_group.add_argument("--seed", type=int, default=666, help="Seed used for the environment")
    env_group.add_argument("--max_episode_length_s", type=int, default=15, help='reference length')
    env_group.add_argument("--task", type=str, default='Go2Ar_Go2Ar', help="Name of the task.")


    # TM2 motion setting
    env_group.add_argument("--motion_name", type=str, default='sparring', help="train motion name")
    env_group.add_argument("--motion_length", type=int, default=900, help='reference length')
    env_group.add_argument("--ref_horizon", type=int, default=30, help='reference length')


    # training setting
    train_group = parser.add_argument_group("Training", description="Arguments for Training Setting.")
    train_group.add_argument("--max_iterations", type=int, default=None, help="Max training iteraction")
    train_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    train_group.add_argument("--resume", type=bool, default=None, help="Whether to resume from a checkpoint.")
    train_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    train_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    train_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb",}, help="Logger module to use."
    )

    return parser.parse_args()



def parse_env_cfg(args):
    from TM2_buddyImitation.configs.env_cfg import TM2Cfg
    
    env_cfg = TM2Cfg()
    if args.device is not None:
        env_cfg.sim_params.device = args.device
    if args.num_envs is not None:
        env_cfg.env.num_envs = args.num_envs
    if args.max_episode_length_s is not None:
        env_cfg.env.max_episode_length_s = args.max_episode_length_s

    
    env_cfg.robot_setting = args.task
    env_cfg.demo.demo_name = args.motion_name
    env_cfg.demo.data_dir = TM2_ROOT_DIR+'/TM2_buddyImitation/results/saved/trajectories/dataset/{}.npy'.format(env_cfg.demo.demo_name)
    env_cfg.seed = args.seed
    env_cfg.sim_params.render = args.render
    
    return env_cfg



def parse_train_cfg(args):

    from TM2_buddyImitation.configs.train_cfg import BuddyImitationCfgPPO
    train_cfg = BuddyImitationCfgPPO()

    if args.seed is not None:
        train_cfg.seed = args.seed
    if args.max_iterations is not None:
        train_cfg.max_iterations = args.max_iterations
    if args.resume:
        train_cfg.resume = args.resume
    if args.run_name is not None:
        train_cfg.run_name = args.run_name
    if args.load_run is not None:
        train_cfg.load_run = args.load_run
    if args.checkpoint is not None:
        train_cfg.checkpoint = args.checkpoint
    if args.logger is not None:
        train_cfg.logger = args.logger

    train_cfg.experiment_name=args.task
    return train_cfg



def get_log_dir(env_cfg, train_cfg, args, log_root=None):
    # specify directory for logging experiments    
    if log_root is None:
        log_root_path = os.path.join(TM2_RESULT_DIR,  args.task, args.motion_name)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    if not train_cfg.run_name:
        train_cfg.run_name = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(log_root_path, train_cfg.run_name)

    return log_root_path, log_dir



def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_env(env_cfg, args):
    from TM2_buddyImitation.envs.TM2_env import TM2_Env, TM2Simple_Env
    if args.task == 'Go2Ar':
        env = TM2Simple_Env(env_cfg, args.render)
    else:
        env = TM2_Env(env_cfg, args.render)
    return env

def get_checkpoint_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        #TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run==-1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    return load_path


def load_runner_run(runner, log_root_path, train_cfg, device):
    # get path to previous checkpoint
    resume_path = get_checkpoint_path(log_root_path, train_cfg.load_run, train_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    runner.load(resume_path, device=device)
    return resume_path


def load_primitives(runner, log_root_path, agent_cfg, source_key, target_key, update_primitive=True):
    
    # get path to previous checkpoint
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    runner.load_primitives(resume_path,  source_key, target_key, update_primitive=update_primitive)
    return resume_path
