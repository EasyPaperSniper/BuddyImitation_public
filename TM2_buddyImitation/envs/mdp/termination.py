import torch

class BaseTerminationManager():

    def check_termination(self, env):
        env.reset_buf = torch.any(torch.norm(env.rob_contact_forces[:, env.termination_contact_indices, :], dim=-1) > 1., dim=1)
        env.time_out_buf = env.episode_length_buf >= env.max_episode_length # no terminal reward for time-outs
        env.reset_buf |= env.time_out_buf
        
        return env.reset_buf, env.time_out_buf
    




class TM2SimpleTerminationManager(BaseTerminationManager):

    def check_termination(self, env):
        env.reset_buf, env.time_out_buf = super().check_termination(env)        
        env.reset_buf |= env.delta_root_c1_pos > 5. #0.3
        # env.reset_buf |= env.delta_root_c1_ori > 2 #1
        env.reset_buf |= (torch.abs(env.rob_proj_gravity[:,-1])<0.95)
        env.reset_buf |= env.rob_root_states[:,2]< env.cfg.rewards.base_height_target* 0.5 # base height too low
        env.reset_buf |= env.time_out_buf
        
        return env.reset_buf, env.time_out_buf


class TM2TerminationManager(BaseTerminationManager):

    def check_termination(self, env):
        env.reset_buf, env.time_out_buf = super().check_termination(env)
        
        env.time_out_buf |= torch.any(torch.abs(env.rob_root_states_r2[:,0:2]) > 180., dim=-1) 
        env.reset_buf |= (torch.abs(env.rob_proj_gravity_r2[:,-1])<0.75)

        env.reset_buf |= env.time_out_buf
        
        return env.reset_buf, env.time_out_buf
