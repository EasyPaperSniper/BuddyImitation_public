import torch
from TM2_buddyImitation.utils.math import euler_xyz_from_quat

class BaseObsManager():

    def compute_observation(self, env):
        env.obs_buf = torch.cat((  env.rob_base_lin_vel * env.obs_scales.lin_vel,
                                    env.rob_base_ang_vel * env.obs_scales.ang_vel,
                                    env.rob_proj_gravity,
                                    env.commands,
                                    env.dof_pos,
                                    env.dof_vel * env.obs_scales.dof_vel,
                                    env.actions
                                    ),dim=-1)
        env.privileged_obs_buf =  torch.cat((  env.rob_base_lin_vel * env.obs_scales.lin_vel,
                                    env.rob_base_ang_vel * env.obs_scales.ang_vel,
                                    env.rob_proj_gravity,
                                    env.commands,
                                    env.dof_pos ,
                                    env.dof_vel * env.obs_scales.dof_vel,
                                    env.actions
                                    ),dim=-1)
        clip_obs = env.cfg.normalization.clip_observations
        env.obs_buf = torch.clip(env.obs_buf, -clip_obs, clip_obs)
        env.privileged_obs_buf = torch.clip(env.privileged_obs_buf, -clip_obs, clip_obs)
        return env.obs_buf,  env.privileged_obs_buf
    


class TM2SimpleObsManager(BaseObsManager):
    def __init__(self, env):
        # Initialize the observation buffer
        self.compute_observation(env)

    def compute_observation(self, env):
        
        # global
        oracle_obs = torch.cat((
            env.cur_ref_pos_r1,
            env.cur_ref_ori_r1,
            env.rob_root_states[:,0:3] - env.env_origins,
            env.rob_root_states[:,3:]
            ), dim=-1)


        # r1 senseor
        r1_prop_obs = torch.cat((
                    env.rob_proj_gravity,
                    env.dof_pos,
                    env.dof_vel,
                    env.last_actions[:,:env.num_actions_r1]
                ), dim=-1)

      
        env.obs_buf = torch.cat((
                                    oracle_obs,
                                    r1_prop_obs,
                                    ),dim=-1)
        

        if not hasattr(env, 'num_obs_list'):
            env.num_obs_list = [oracle_obs.shape[-1], r1_prop_obs.shape[-1]]
            env.num_actions_list = [env.num_actions_r1]


        env.privileged_obs_buf = env.obs_buf.clone()
        clip_obs = env.cfg.normalization.clip_observations
        env.obs_buf = torch.clip(env.obs_buf, -clip_obs, clip_obs)
        env.privileged_obs_buf = torch.clip(env.privileged_obs_buf, -clip_obs, clip_obs)



        return env.obs_buf,  env.privileged_obs_buf




class TM2ObsManager(TM2SimpleObsManager):
    def __init__(self, env):
        self.compute_observation(env)

    def compute_observation(self, env):
        
        # global
        oracle_obs = torch.cat((
            env.cur_ref_motion_r1,
            env.rob_root_states[:,0:13],
            env.cur_ref_motion_r2,
            env.rob_root_states_r2[:,0:13],
            ), dim=-1)


        # r1 senseor
        r1_prop_obs = torch.cat((
                    env.rob_proj_gravity,
                    env.dof_pos,
                    env.dof_vel,
                    env.actions[:,:env.num_actions_r1]
                ), dim=-1)

        # r2 senseor
        r2_prop_obs = torch.cat((
                    env.rob_proj_gravity_r2,
                    env.dof_pos_r2,
                    env.dof_vel_r2,
                    env.actions[:,env.num_actions_r1:]
                ), dim=-1)
        
        # env.
      
        env.obs_buf = torch.cat((
                                    oracle_obs,
                                    r1_prop_obs,
                                    r2_prop_obs,
                                    ),dim=-1)

        env.privileged_obs_buf = env.obs_buf.clone()
        clip_obs = env.cfg.normalization.clip_observations
        env.obs_buf = torch.clip(env.obs_buf, -clip_obs, clip_obs)
        env.privileged_obs_buf = torch.clip(env.privileged_obs_buf, -clip_obs, clip_obs)



        return env.obs_buf,  env.privileged_obs_buf