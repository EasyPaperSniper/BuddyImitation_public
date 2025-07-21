import torch
import numpy as np
import genesis as gs
import genesis.utils.geom as gu


from TM2_buddyImitation import TM2_ENVS_DIR, TM2_ROOT_DIR
from TM2_buddyImitation.envs.assets import *
from TM2_buddyImitation.envs.mdp import *
from TM2_buddyImitation.utils.math import quat_rotate_inverse, yaw_quat, quat_from_euler_xyz
from TM2_buddyImitation.envs.base_env import BaseEnv




class TM2Simple_Env(BaseEnv):
    # for baseline
    def __init__(self, env_cfg, render=False):
        super().__init__(env_cfg, render)


    def _init_managers(self,):
        self.reset_manager = TM2SimpleResetManager()
        self.obs_manager = TM2SimpleObsManager(self)
        self.action_manager = TM2SimpleActionManager()
        self.reward_manager = TM2SimpleRewardManager(self)
        self.termination_manager = TM2SimpleTerminationManager()
        self.command_manager = TM2SimpleCommandManager(self.cfg.command)
        self._post_init_managers()


    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs = self.compute_observations()
        return obs, privileged_obs

    def reset_idx(self,env_ids, ref_index=None):
        """ Reset all robots with given env_ids"""
        self.reset_manager.reset_idx(self, env_ids, ref_index)
        self._update_states()
        self.sample_commands()


    def step(self, actions):
        action_r1  = self.action_manager.compute(actions, self)
        for _ in range(self.ctrl_decimation):
            self.robot.control_dofs_position(action_r1, np.arange(self.num_actions_r1)+6)
            self.scene.step()
        self.post_physics_step()
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def _load_terrain(self):
        self.plane = self.scene.add_entity(gs.morphs.Plane(),)

    def _load_entities(self):
        self.robot = self._load_invdividual_entity(type =self.cfg.robot.type,
                                        file_dir=self.cfg.robot.file_dir,
                                        pos=(0,0,1))
        

    def _update_states(self,):
        super()._update_states()
        
        # robot_1 state
        self.rob_heading_quat = yaw_quat(self.rob_base_quat)
        self.rob_base_lin_vel = quat_rotate_inverse(self.rob_heading_quat, self.rob_root_states[:, 7:10])
        self.rob_base_ang_vel = quat_rotate_inverse(self.rob_heading_quat, self.rob_root_states[:, 10:13])
        self.update_cur_ref()


    def _init_env_vectors(self):
        self.load_ref_data(self.cfg.demo.data_dir)
        super()._init_env_vectors()
        self.num_actions_r1 =  self.motor_dof.shape[0]
        


    def load_ref_data(self, data_dir):
        self.ref_data = torch.from_numpy(np.load(data_dir)).to(device=self.device)
        if self.ref_data.ndim == 4:
            self.ref_data = self.ref_data.reshape(-1, self.ref_data.shape[-2], self.ref_data.shape[-1])
        self.ref_index_r1 = torch.randint(0, self.ref_data.shape[0], size=(self.num_envs,), device=self.device, requires_grad=False)
        self.ref_data_r1 = self.ref_data[self.ref_index_r1]

        self.rob_default_qpos[:,0:2] = self.ref_data_r1[:,0,66:66+2]
        self.rob_default_qpos[:,3:7] = yaw_quat(self.ref_data_r1[:,0,69:69+4])



    def update_cur_ref(self):
        ref_idx = self.episode_length_buf.unsqueeze(-1)//3 + torch.tensor([0, 2, self.cfg.demo.ref_horizon], device=self.device)
        self.cur_ref_pos_r1 = (self.ref_data_r1[torch.arange(self.num_envs, device=self.device).unsqueeze(1),ref_idx])[...,0:3].reshape(self.num_envs,-1)
        self.cur_ref_ori_r1 = (self.ref_data_r1[torch.arange(self.num_envs, device=self.device).unsqueeze(1),ref_idx])[...,69:73].reshape(self.num_envs,-1)
        self.cur_ref_root_ori_r1 = yaw_quat(self.cur_ref_ori_r1[:, 0:4])
        
        self.delta_root_c1_pos = torch.norm(self.cur_ref_pos_r1[:,0:2]  - self.rob_root_states[:,0:2] + self.env_origins[:,0:2], dim=-1)
        self.delta_root_c1_ori = quat_error_magnitude(self.cur_ref_root_ori_r1, self.rob_root_states[:,3:7])





class TM2_Env(BaseEnv):
    def __init__(self, env_cfg, render=False):
        super().__init__(env_cfg, render)


    def _init_managers(self,):
        self.reset_manager = TM2ResetManager()
        self.obs_manager = TM2ObsManager(self)
        self.action_manager = TM2ActionManager()
        self.reward_manager = TM2RewardManager(self)
        self.termination_manager = TM2TerminationManager()
        self.command_manager = TM2CommandManager(self.cfg.command)
        self._post_init_managers()


    def step(self, actions):
        action_r1, action_r2 = self.action_manager.compute(actions, self)
        for _ in range(self.ctrl_decimation):
            self.robot.control_dofs_position(action_r1, np.arange(self.num_actions_r1)+6)
            self.robot_2.control_dofs_position(action_r2, np.arange(self.num_actions_r2)+6)
            self.scene.step()
        self.post_physics_step()
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    

   
      
    def _create_scene(self):
        super()._create_scene()

        self.motor_dof_r2 =  torch.cat([torch.tensor(self.robot_2.get_joint(name).dofs_idx,dtype=torch.long, device=self.device, requires_grad=False) 
                                                for name in self.cfg.robot_2.robot_motor_name] )
        self.arm_motor_dof_r2 = torch.cat([torch.tensor(self.robot_2.get_joint(name).dofs_idx,dtype=torch.long, device=self.device, requires_grad=False) 
                                                for name in self.cfg.robot_2.robot_arm_motor_name] )
        self.leg_motor_dof_r2 = torch.cat([torch.tensor(self.robot_2.get_joint(name).dofs_idx,dtype=torch.long, device=self.device, requires_grad=False) 
                                                for name in self.cfg.robot_2.robot_leg_motor_name] )
        self.feet_indices_r2 = torch.tensor([self.robot_2.get_link(name).idx for name in self.cfg.robot_2.robot_foot_name],
                                                                    dtype=torch.long, device=self.device, requires_grad=False)-self.robot_2.link_start
        self.penalised_contact_indices_r2 = torch.tensor([self.robot_2.get_link(name).idx for name in self.cfg.robot_2.penalize_contacts_on],
                                                                    dtype=torch.long, device=self.device, requires_grad=False)-self.robot_2.link_start
        self.termination_contact_indices_r2 = torch.tensor([self.robot_2.get_link(name).idx for name in self.cfg.robot_2.terminate_after_contacts_on],
                                                                    dtype=torch.long, device=self.device, requires_grad=False)-self.robot_2.link_start
        

  
        motor_dof_local_r2 = self.motor_dof_r2 - self.robot_2.dof_start
        rob_qs_idx_local_r2 = torch.cat([torch.arange(7, device=self.device, requires_grad=False), motor_dof_local_r2 + 1])
        self.robot_2.set_qpos(torch.tensor(self.cfg.robot_2.robot_default_qpos, device=self.device).repeat((self.num_envs,1)), qs_idx_local=rob_qs_idx_local_r2)
        self.rob_default_qpos_r2 = self.robot_2.get_qpos().clone()
        self.rob_default_dof_pos_r2 = self.rob_default_qpos[:,7:]


        self.robot_2.set_dofs_kp(self.cfg.robot_2.motor_kp, motor_dof_local_r2)
        self.robot_2.set_dofs_kv(self.cfg.robot_2.motor_kv, motor_dof_local_r2)
        self.rob_kp_r2 = self.robot_2.get_dofs_kp()
        self.rob_kv_r2 = self.robot_2.get_dofs_kv()
        self.rob_dof_pos_limits_r2 = self.robot_2.get_dofs_limit()
        self.rob_dof_force_limits_r2 = self.robot_2.get_dofs_force_range()


    
    def _load_entities(self):
        super()._load_entities()
        self.robot_2 = self._load_invdividual_entity(type =self.cfg.robot_2.type,
                                        file_dir=self.cfg.robot_2.file_dir,
                                        pos=(1,1,1))



    def _init_env_vectors(self):
        # self.aig_idx = torch.tensor(self.cfg.robot.aig_idx).to(device=self.device)
        self.rob_root_states_r2 = torch.zeros(self.num_envs, 13, device=self.device, dtype=torch.float)
            
        super()._init_env_vectors()

        self.num_actions_r2 =  self.motor_dof_r2.shape[0]
        self.num_actions =  self.num_actions_r1 + self.num_actions_r2
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = self.actions[:].clone()


        self.last_contacts_r2 = torch.zeros(self.num_envs, len(self.feet_indices_r2), dtype=torch.bool, device=self.device, requires_grad=False)
        self.feet_air_time_r2 = torch.zeros(self.num_envs, self.feet_indices_r2.shape[0], dtype=torch.float, device=self.device, requires_grad=False)

        
        
        
    def _update_states(self,):
        super()._update_states()


        # robot_2 state
        rob_qpos_r2 = self.robot_2.get_qpos()
        rob_vel_r2 = self.robot_2.get_dofs_velocity()
        self.rob_root_states_r2[:,0:7] = rob_qpos_r2[:,0:7]
        self.rob_root_states_r2[:,7:] = rob_vel_r2[:,0:6]
        self.rob_base_quat_r2 = self.rob_root_states_r2[:, 3:7]
        self.dof_pos_r2 = rob_qpos_r2[:,7:]
        self.dof_vel_r2 = rob_vel_r2[:,6:]
        self.rob_proj_gravity_r2 = quat_rotate_inverse(self.rob_base_quat_r2, self.gravity_vec)
        self.rob_heading_quat_r2 = yaw_quat(self.rob_base_quat_r2)
        self.rob_base_lin_vel_r2 = quat_rotate_inverse(self.rob_heading_quat_r2, self.rob_root_states_r2[:, 7:10])
        self.rob_base_ang_vel_r2 = quat_rotate_inverse(self.rob_heading_quat_r2, self.rob_root_states_r2[:, 10:13])
        self.rob_contact_forces_r2 = self.robot_2.get_links_net_contact_force().clone().detach()

        self.update_cur_ref()


    def load_ref_data(self, data_dir):
        super().load_ref_data(data_dir)
        self.rob_default_qpos_r2[:,0:2] = self.ref_data_r2[0,66:66+2]
        self.rob_default_qpos_r2[:,3:7] = yaw_quat(self.ref_data_r2[0,69:69+4])




