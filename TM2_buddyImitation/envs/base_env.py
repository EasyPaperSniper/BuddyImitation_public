import torch
import numpy as np
import genesis as gs
import genesis.utils.geom as gu

from TM2_buddyImitation.envs.assets import *
from TM2_buddyImitation.envs.mdp import *
from TM2_buddyImitation.utils.helpers import class_to_dict
from TM2_buddyImitation.utils.math import quat_rotate_inverse


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class BaseEnv:
    '''
    Locomotion Env with Flat Terrain
    '''
    def __init__(self, env_cfg, render=False) -> None:
        self.cfg = env_cfg
        self.render = render
        self._parse_cfg()
        self._create_scene()
        self._init_env_vectors()
        self._init_managers()
        
    
    def get_observations(self):
        return self.obs_buf
    
    def get_privileged_observations(self):
        return self.privileged_obs_buf
    
    def step(self, actions):
        tgt_pose = self.action_manager.compute(actions, self)
        for _ in range(self.ctrl_decimation):
            self.robot.control_dofs_position(tgt_pose, np.arange(self.num_actions)+6)
            self.scene.step()
        self.post_physics_step()
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs = self.compute_observations()
        return obs, privileged_obs
    
    def set_camera_pose(self, pos=None, lookat=None):
        if not self.render:
            return
        self.scene.viewer.set_camera_pose(pos, lookat)

    def post_physics_step(self):
        self.episode_length_buf += 1

        # prepare quantities
        self._update_states()
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:].clone()
        self.last_dof_vel[:] = self.dof_vel[:].clone()
        self.last_root_vel[:] = self.rob_root_states[:, 7:13].clone()
        

    def reset_idx(self,env_ids):
        self.reset_manager.reset_idx(self, env_ids)
        self._update_states()
        self.sample_commands()
        
    def compute_observations(self):
        self.obs_buf, self.privileged_obs_buf = self.obs_manager.compute_observation(self)
        return self.obs_buf, self.privileged_obs_buf
        
    def compute_reward(self):
        self.reward_manager.compute(self)

    def check_termination(self):
        self.reset_buf, self.time_out_buf = self.termination_manager.check_termination(self)

    def sample_commands(self):
        self.commands = self.command_manager.sample_commands(self)
        


    # =================== callbacks =========================== #    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
    
        self.sample_commands()


    # ++++++++++++++++++ Env Setup ++++++++++++++++++++++++++ #
    def _create_scene(self):
        gs.init(seed=self.cfg.seed,
                backend=gs.cpu if self.device=='cpu' else gs.gpu,
                debug=False,
                logging_level='warning')

        self.scene = gs.Scene(
                sim_options=self.gs_SimOption,
                viewer_options=self.gs_ViewerOptions,
                rigid_options=self.gs_RigidOption,
                show_viewer=self.render,
            )
        self._load_terrain()
        self._load_entities()
        self.scene.build(n_envs=self.num_envs)

        self.motor_dof = torch.cat([torch.tensor(self.robot.get_joint(name).dofs_idx,dtype=torch.long, device=self.device, requires_grad=False) 
                                                for name in self.cfg.robot.robot_motor_name] )
        self.arm_motor_dof = torch.cat([torch.tensor(self.robot.get_joint(name).dofs_idx,dtype=torch.long, device=self.device, requires_grad=False)
                                         for name in self.cfg.robot.robot_arm_motor_name])
        self.leg_motor_dof = torch.cat([torch.tensor(self.robot.get_joint(name).dofs_idx,dtype=torch.long, device=self.device, requires_grad=False)
                                         for name in self.cfg.robot.robot_leg_motor_name])
        

        self.feet_indices = torch.tensor([self.robot.get_link(name).idx for name in self.cfg.robot.robot_foot_name],
                                                                    dtype=torch.long, device=self.device, requires_grad=False)-self.robot.link_start
        self.penalised_contact_indices = torch.tensor([self.robot.get_link(name).idx for name in self.cfg.robot.penalize_contacts_on],
                                                                    dtype=torch.long, device=self.device, requires_grad=False)-self.robot.link_start
        self.termination_contact_indices = torch.tensor([self.robot.get_link(name).idx for name in self.cfg.robot.terminate_after_contacts_on],
                                                                    dtype=torch.long, device=self.device, requires_grad=False)-self.robot.link_start
        
        rob_qs_idx = torch.cat([torch.arange(7, device=self.device, requires_grad=False), self.motor_dof+1])
        self.robot.set_qpos(torch.tensor(self.cfg.robot.robot_default_qpos, device=self.device).repeat((self.num_envs,1)), qs_idx_local=rob_qs_idx)
        self.rob_default_qpos = self.robot.get_qpos().clone()
        self.rob_default_dof_pos = self.rob_default_qpos[:,7:]


        self.robot.set_dofs_kp(self.cfg.robot.motor_kp, self.motor_dof) 
        self.robot.set_dofs_kv(self.cfg.robot.motor_kv, self.motor_dof)
        self.rob_kp = self.robot.get_dofs_kp()
        self.rob_kv = self.robot.get_dofs_kv()
        self.rob_dof_pos_limits = self.robot.get_dofs_limit()
        self.rob_dof_force_limits = self.robot.get_dofs_force_range()


    def _load_terrain(self):
        self.plane = self.scene.add_entity(
                gs.morphs.URDF(file=self.cfg.terrain.terrain_dir, fixed=True)
                        )
        self.plane.set_friction(self.cfg.terrain.static_friction)


    def _load_entities(self):
        self.robot = self._load_invdividual_entity(type =self.cfg.robot.type,
                                        file_dir=self.cfg.robot.file_dir,
                                        pos=self.cfg.robot.robot_default_qpos[0:3])
        


    def _load_invdividual_entity(self, type, file_dir, pos, convexify=True, **kwargs):
        if type == 'URDF':
            entity = self.scene.add_entity(
                    gs.morphs.URDF(file=file_dir,
                                pos=pos,
                                convexify=convexify,
                                merge_fixed_links=False,
                                ),
                )
        elif type == 'MJCF':
            entity = self.scene.add_entity(
                    gs.morphs.MJCF(file=file_dir,
                                pos=pos,
                                convexify=convexify,
                                ),
                )
        else:
            raise Exception("Type not Implemented")
        
        return entity


    def _parse_cfg(self):
        self.sim_params = self.cfg.sim_params
        self.device = self.cfg.sim_params.device
        self.sim_dt = self.cfg.sim_params.sim_dt
        self.ctrl_decimation = self.cfg.sim_params.decimation
        self.ctrl_dt = self.sim_dt * self.ctrl_decimation
        self.dt = self.ctrl_dt
        
        self.num_envs = self.cfg.env.num_envs
        self.max_episode_length_s = self.cfg.env.max_episode_length_s
        self.max_episode_length = int(self.max_episode_length_s//self.ctrl_dt)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device)
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.obs_scales = self.cfg.normalization.obs_scales

        self._prepare_genesis_options()


    def _prepare_genesis_options(self,):
        self.gs_SimOption = gs.options.SimOptions(
                                    dt = self.cfg.sim_params.sim_dt,
                                    )
        
        self.gs_RigidOption = gs.options.RigidOptions(
                    enable_collision=True,
                    enable_joint_limit=True,
                    constraint_solver=gs.constraint_solver.Newton,
                )


        self.gs_ViewerOptions = gs.options.ViewerOptions(
                res=(1280, 960),
                camera_pos=self.cfg.viewer.pos,
                camera_lookat=self.cfg.viewer.lookat,
                camera_fov=40,
            )


    def _get_env_origins(self):
        self.env_origins = torch.zeros(self.num_envs,3, device=self.device)
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        spacing = self.cfg.env.env_spacing
        xx, yy = torch.meshgrid(torch.arange(-num_rows,num_rows,2), torch.arange(-num_cols, num_cols,2))
        spacing = self.cfg.env.env_spacing/2
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.


    def _init_env_vectors(self):
        self.num_actions = self.motor_dof.shape[0]
        self.rob_root_states = torch.zeros(self.num_envs, 13, device=self.device, dtype=torch.float)
        self.gravity_vec = torch.tensor([0., 0., 1.], device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = torch.tensor([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))

        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self._get_env_origins()
        self._update_states()


        self.commands = torch.zeros((self.num_envs, self.cfg.command.cmd_dof),device=self.device, dtype=torch.float, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = self.actions[:]
        self.last_dof_vel = self.dof_vel[:]
        self.last_root_vel = self.rob_root_states[:, 7:13]
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices),  device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)

       

       
    def _update_states(self,):
        rob_qpos = self.robot.get_qpos()
        rob_vel = self.robot.get_dofs_velocity()
        
        self.rob_root_states[:,0:7] = rob_qpos[:,0:7]
        self.rob_root_states[:,7:] = rob_vel[:,0:6]
        self.rob_base_quat = self.rob_root_states[:, 3:7]
        self.dof_pos = rob_qpos[:,7:]
        self.dof_vel = rob_vel[:,6:]

        self.rob_base_lin_vel = quat_rotate_inverse(self.rob_base_quat, self.rob_root_states[:, 7:10])
        self.rob_base_ang_vel = quat_rotate_inverse(self.rob_base_quat, self.rob_root_states[:, 10:13])
        self.rob_proj_gravity = quat_rotate_inverse(self.rob_base_quat, self.gravity_vec)

        self.rob_contact_forces = self.robot.get_links_net_contact_force().clone().detach()
        self.rob_foot_contact = self.rob_contact_forces[:, self.feet_indices, 2] > 1.



    def _init_managers(self,):
        self.reset_manager = BaseResetManager()
        self.obs_manager = BaseObsManager()
        self.action_manager = BaseActionManager()
        self.reward_manager = BaseRewardManager(self)
        self.termination_manager = BaseTerminationManager()
        self.command_manager = BaseCommandManager(self.cfg.command)

        self._post_init_managers()


    def _post_init_managers(self):
        self.obs_buf, self.privileged_obs_buf = self.obs_manager.compute_observation(self)
        self.num_obs = self.obs_buf.shape[-1]
        if self.privileged_obs_buf is not None:
            self.num_privileged_obs = self.privileged_obs_buf.shape[-1]
        self.extras = {}
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

        self.reset_idx(range(self.num_envs))