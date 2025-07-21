import time
import os
from collections import deque
import statistics

import wandb
import torch
import torch.nn as nn


from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic, EmpiricalNormalization


from TM2_buddyImitation.utils.wandb_utils import NoneSummaryWrite
from TM2_buddyImitation.runner.learning_module import TM2_ActorCritic




class Interaction_Transfer(OnPolicyRunner):
    def __init__(self,
                 env,
                 train_cfg,
                 log_dir=None,
                 ):

        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.empirical_normalization = train_cfg["empirical_normalization"]
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        self.device = env.device
        self.env = env


        num_actor_obs_list = self.env.num_obs_list
        num_critic_obs = self.env.num_privileged_obs
        num_actions_list = self.env.num_actions_list
        num_actor_obs = self.env.num_obs
        num_actions = self.env.num_actions

        actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # ActorCritic
        actor_critic =  actor_critic_class(
                    num_actor_obs_list, num_critic_obs, num_actions_list, **self.policy_cfg).to(self.device)
        


        alg_class = eval(self.alg_cfg.pop("class_name"))  # PPO
        self.alg = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        

        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_actor_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity()  # no normalization


        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_actor_obs],
            [num_critic_obs],
            [num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = NoneSummaryWrite()
        self.logger_type = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0



        
    def learn(self, num_learning_iterations):
        # initialize writer
        if self.log_dir is not None:
            print(f"########Logging to {self.log_dir}########")
            os.makedirs(self.log_dir, exist_ok=True)

        if self.log_dir is not None  and self.cfg["logger"]=='wandb':
            from TM2_buddyImitation.utils.wandb_utils import WandbSummaryWriter
            self.writer = WandbSummaryWriter(self.cfg, self.log_dir)
            self.logger_type = 'wandb'
        
        obs, critic_obs = self.env.compute_observations()
        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter+1, tot_iter+1):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range( self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, critic_obs, rewards, dones, infos = self.env.step(actions)
                    obs = self.obs_normalizer(obs)
                    critic_obs = self.critic_obs_normalizer(critic_obs)

                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))



    def save(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
        torch.save(saved_dict, path)


    def load(self, path, load_optimizer=True, device=None):
        loaded_dict = torch.load(path, map_location=device)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]
    


    def load_primitives(self, path,  source_key, target_key, update_primitive=True, device=None):
        if device is None:
            device = self.env.device
        loaded_dict = torch.load(path, map_location=device)
        for name, param in loaded_dict.items():
            if source_key in name:
                full_tgt_key = name.replace(source_key, target_key)
                self.alg.actor_critic.state_dict()[full_tgt_key].copy_(param)
                if not update_primitive:
                    for param in  self.alg.actor_critic.state_dict()[full_tgt_key].parameters():
                        param.requires_grad = False
       
