
from TM2_buddyImitation import TM2_ENVS_DIR, TM2_ROOT_DIR
from TM2_buddyImitation.envs.assets import *

from TM2_buddyImitation.utils.configclass import BaseConfig




class TM2Cfg(BaseConfig):
    
    robot_setting = 'Go2Ar_Go2Ar'
    robot = ASSETS_CFG_DICT[robot_setting.split('_')[0]]
    robot_2 = ASSETS_CFG_DICT[robot_setting.split('_')[1]]

    seed = 0


    class sim_params:
        device = 'mps:0'
        sim_dt = 1/120
        decimation = 2
        render = False


    class env:
        num_envs = 4
        env_spacing = 2
        max_episode_length_s = 10
        send_timeouts = True


    class demo:
        demo_name = 'sparring'
        data_dir = TM2_ROOT_DIR+'/TM2_buddyImitation/results/saved/trajectories/dataset/{}.npy'.format(demo_name)
        EIG_dir = TM2_ROOT_DIR+'/TM2_buddyImitation/results/saved/trajectories/dataset/{}_EIG.npy'.format(demo_name)
        ref_horizon = 20

    
    class rewards:
        class scales:
            tracking_graph = 1.5 #1
            base_height = -0.5
            feet_air_time = -1.
            action_rate = -0.05 # -0.02
            action = -0.02 #-0.01
            collision = -0.1
            joint_reg = -0.1
            dof_pos_limits = -1
            hip_joint_reg = -1


        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.5 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = .9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = .9
        soft_torque_limit = 1.
        base_height_target = 0.32
        max_contact_force = 100. # forces above this value are penalized


    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.



    class command:
        cmd_dof = 3
        cmd_interval_s = None




    class viewer:
        pos = [-1, -3, 2]  # [m]
        lookat = [0., 0, 1.]  # [m]



