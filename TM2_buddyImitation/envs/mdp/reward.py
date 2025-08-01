import torch
from TM2_buddyImitation.utils.math import quat_error_magnitude,quat_rotate_inverse, yaw_quat, quat_from_euler_xyz


class BaseRewardManager():
    def __init__(self, env) -> None:
        self._prepare_reward_function(env)


    def compute(self, env):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in env_prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        env.rew_buf[:] = 0.
        for i in range(len(env.reward_functions)):
            name = env.reward_names[i]
            rew = env.reward_functions[i](env) * env.reward_scales[name]
            env.rew_buf += rew
            env.episode_sums[name] += rew
        if env.cfg.rewards.only_positive_rewards:
            env.rew_buf[:] = torch.clip(env.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in env.reward_scales:
            rew = env._reward_termination(env) * env.reward_scales["termination"]
            env.rew_buf += rew
            env.repisode_sums["termination"] += rew
        return env.rew_buf, env.episode_sums
        

    def _prepare_reward_function(self, env):
        # remove zero scales + multiply non-zero ones by dt
        for key in list(env.reward_scales.keys()):
            scale = env.reward_scales[key]
            if scale==0:
                env.reward_scales.pop(key) 
            else:
                env.reward_scales[key] *= env.dt
        # prepare list of functions
        env.reward_functions = []
        env.reward_names = []
        for name, scale in env.reward_scales.items():
            if name=="termination":
                continue
            env.reward_names.append(name)
            name = '_reward_' + name
            env.reward_functions.append(getattr(self, name))


    # =================== reward functions ==================== #
    def _reward_lin_vel_z(self, env):
        # Penalize z axis base linear velocity
        return torch.square(env.rob_base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self, env):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(env.rob_base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self, env):
        # Penalize non flat base orientation
        return torch.sum(torch.square(env.rob_proj_gravity[:, :2]), dim=1)

    def _reward_base_height(self, env):
        # Penalize base height away from target
        return torch.abs(env.rob_root_states[:, 2] - env.cfg.rewards.base_height_target)
    
    def _reward_torques(self, env):
        # Penalize torques
        return torch.sum(torch.square(env.torques), dim=1)

    def _reward_dof_vel(self, env):
        # Penalize dof velocities
        return torch.sum(torch.square(env.dof_vel), dim=1)
    
    def _reward_dof_acc(self, env):
        # Penalize dof accelerations
        return torch.sum(torch.square((env.last_dof_vel - env.dof_vel) / env.dt), dim=1)
    
    def _reward_action_rate(self, env):
        # Penalize changes in actions
        return torch.sum(torch.square(env.last_actions - env.actions), dim=1)
    
    def _reward_collision(self, env):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(env.rob_contact_forces[:, env.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self, env):
        # Terminal reward / penalty
        return env.reset_buf * ~env.time_out_buf
    
    def _reward_dof_pos_limits(self, env):
        # Penalize dof positions too close to the limit
        out_of_limits = -(env.dof_pos - env.rob_dof_pos_limits[0][6:]).clip(max=0.) # lower limit
        out_of_limits += (env.dof_pos - env.rob_dof_pos_limits[1][6:]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self,env):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(env.dof_vel) - env.dof_vel_limits*env.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self, env):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(env.torques) - env.torque_limits*env.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self, env):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(env.commands[:, :2] - env.rob_base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/env.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self, env):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(env.commands[:, 2] - env.rob_base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/env.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self, env):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = env.rob_contact_forces[:, env.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, env.last_contacts) 
        env.last_contacts = contact
        first_contact = (env.feet_air_time > 0.) * contact_filt
        env.feet_air_time += env.dt
        rew_airTime = torch.sum((env.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(env.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        env.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self, env):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(env.rob_contact_forces[:, env.feet_indices, :2], dim=2) >\
             5 *torch.abs(env.rob_contact_forces[:, env.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self, env):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(env.dof_pos - env.rob_default_dof_pos), dim=1) * (torch.norm(env.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self, env):
        # penalize high contact forces
        return torch.sum((torch.norm(env.rob_contact_forces[:, env.feet_indices, :], dim=-1) -  env.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_arm_movement(self, env):
        return torch.sum(torch.square(env.dof_pos[:,env.arm_motor_dof-6] - env.rob_default_dof_pos[:,env.arm_motor_dof-6]), dim=1)




class TM2SimpleRewardManager(BaseRewardManager):
    def __init__(self, env) -> None:
        super().__init__(env)
        

    ###############  main rewards  ##########################
    def _reward_tracking_graph(self,env):
        return  0.8*torch.exp(-env.delta_root_c1_pos /env.cfg.rewards.tracking_sigma) + 0.2*torch.exp(-env.delta_root_c1_ori/env.cfg.rewards.tracking_sigma)


    ############## regularization ################################

    def _reward_contact_force(self, env):
        # Penalize collisions on selected bodies
        return torch.norm(env.rob_contact_forces[:,-1], dim=-1)



    def _reward_feet_air_time(self, env):
        # Reward long steps
        contact = env.rob_contact_forces[:, env.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, env.last_contacts) 
        env.last_contacts = contact
        first_contact = (env.feet_air_time > 0.) * contact_filt
        env.feet_air_time += env.dt
        rew_airTime = torch.sum(torch.abs(env.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(env.cur_ref_pos_r1[:,-3:-1]- env.cur_ref_pos_r1[:,:2])>0.1 #no reward for zero command
        env.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_hip_joint_reg(self, env):
        # Penalize hip joint positions
        if not hasattr(env, 'hip_motor_dof'):
            env.hip_motor_dof = torch.cat([torch.tensor(env.robot.get_joint(name).dofs_idx,dtype=torch.long, device=env.device, requires_grad=False)
                                         for name in env.cfg.robot.robot_hip_motor_name])
        return torch.sum(torch.square(env.dof_pos[:, env.hip_motor_dof-6] - env.rob_default_dof_pos[:, env.hip_motor_dof-6]), dim=1)

    def _reward_joint_reg(self, env):
        return torch.sum(torch.abs(env.dof_pos[:,6:] - env.rob_default_dof_pos[:,6:]), dim=1)
    
    # def _reward_action_rate(self, env):
    #     # Penalize changes in actions
    #     return torch.sum(torch.abs(env.last_actions - env.actions), dim=1)
    
    def _reward_action(self, env):
        # Penalize changes in actions
        return torch.sum(torch.square(env.actions), dim=1)







class TM2RewardManager(TM2SimpleRewardManager):

        

    ###############  main rewards  ##########################
    def _reward_tracking_graph(self,env):
        

        return   torch.exp(-torch.norm(env.delta_graph, dim=-1))








   