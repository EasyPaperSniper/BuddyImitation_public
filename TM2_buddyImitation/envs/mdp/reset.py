import torch
from TM2_buddyImitation.utils.math import quat_rotate_inverse, yaw_quat, quat_from_euler_xyz

class BaseResetManager():
    def reset_idx(self, env, env_ids):
        # send timeout info to the algorithm
        if env.cfg.env.send_timeouts:
            env.extras["time_outs"] = env.time_out_buf.clone()

        if len(env_ids) == 0:
            return
        
        # reset robot states
        self._reset_states(env, env_ids)
        self._reset_env_vector(env, env_ids)

       

    def _reset_states(self, env, env_ids):
        reset_qpos = env.rob_default_qpos[env_ids].clone().to(device=env.device)
        reset_qpos[:,0:3] += env.env_origins[env_ids,0:3]
        env.robot.set_qpos(reset_qpos,  zero_velocity=True, envs_idx=env_ids)

        


    def _reset_env_vector(self, env, env_ids):
        # reset buffers        
        env.episode_length_buf[env_ids] = 0
        env.actions[env_ids] = 0
        env.torques[env_ids] = 0
        env.feet_air_time[env_ids] = 0


        # fill extras
        env.extras["episode"] = {}
        for key in env.episode_sums.keys():
            env.extras["episode"]['rew_' + key] = torch.mean(env.episode_sums[key][env_ids]) / env.max_episode_length_s
            env.episode_sums[key][env_ids] = 0.






class TM2SimpleResetManager(BaseResetManager):
    def reset_idx(self, env, env_ids, ref_index=None):
        # send timeout info to the algorithm
        if env.cfg.env.send_timeouts:
            env.extras["time_outs"] = env.time_out_buf.clone()

        if len(env_ids) == 0:
            return
        
        # reset robot states
        self._resample_reference(env, env_ids, ref_index)
        self._reset_states(env, env_ids)
        self._reset_env_vector(env, env_ids)
        


    def _resample_reference(self, env, env_ids, ref_index=None):
        if env.ref_data.ndim == 4:
            env.ref_data = env.ref_data.reshape(-1, env.ref_data.shape[-2], env.ref_data.shape[-1])
        if ref_index is None:
            env.ref_index_r1[env_ids] = torch.randint(0, env.ref_data.shape[0], size=(len(env_ids),), device=env.device, requires_grad=False)
        else:
            env.ref_index_r1[env_ids] = ref_index
        env.ref_data_r1[env_ids] = env.ref_data[env.ref_index_r1[env_ids]]

        env.rob_default_qpos[env_ids,0:2] = env.ref_data_r1[env_ids,0,66:66+2] 
        env.rob_default_qpos[env_ids,3:7] = yaw_quat(env.ref_data_r1[env_ids,0,69:69+4])



class TM2ResetManager(TM2SimpleResetManager):


        
    def _reset_states(self, env, env_ids):
        super()._reset_states(env, env_ids)
        reset_qpos_r2 = env.rob_default_qpos_r2[env_ids].clone().to(device=env.device)
        reset_qpos_r2[:,0:3] += env.env_origins[env_ids,0:3]
        env.robot_2.set_qpos(reset_qpos_r2,  zero_velocity=True, envs_idx=env_ids)


    def _reset_env_vector(self, env, env_ids):
         # reset buffers
        super()._reset_env_vector(env, env_ids)
        env.feet_air_time_r2[env_ids] = torch.zeros(env.feet_air_time_r2[env_ids].shape, device=env.device)
        env.last_contacts_r2[env_ids] = torch.zeros(env.last_contacts_r2[env_ids].shape, device=env.device, dtype=torch.bool, requires_grad=False)
        


            