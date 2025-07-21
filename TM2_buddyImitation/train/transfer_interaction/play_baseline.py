import torch
import numpy as np
from TM2_buddyImitation.runner.TM2_Runners import Interaction_Transfer
import TM2_buddyImitation.utils.helpers as helpers
from TM2_buddyImitation.utils.helpers import class_to_dict



def get_robot_state(env):
    r1_root_states = env.rob_root_states[0,0:7]
    r2_root_states = env.rob_root_states[1,0:7]
    r1_joint_states = torch.cat([env.dof_pos[0,env.leg_motor_dof-6], env.dof_pos[0,env.arm_motor_dof-6]])
    r2_joint_states = torch.cat([env.dof_pos[1,env.leg_motor_dof-6], env.dof_pos[1,env.arm_motor_dof-6]])
    r1_state = torch.cat([r1_root_states, r1_joint_states]).clone().cpu().numpy()
    r2_state = torch.cat([r2_root_states, r2_joint_states]).clone().cpu().numpy()
    return r1_state, r2_state


def test(args):
    env_cfg = helpers.parse_env_cfg(args)
    train_cfg = helpers.parse_train_cfg(args)
    log_root_path, log_dir = helpers.get_log_dir(env_cfg, train_cfg, args)
    
    env = helpers.build_env(env_cfg, args)
    runner = Interaction_Transfer(env, class_to_dict(train_cfg), log_dir=log_dir)

    train_cfg.load_run = 'test'
    train_cfg.load_checkpoint = 'model_200000'
    
    helpers.load_runner_run(runner, log_root_path, train_cfg, env.device)
    helpers.load_primitives(runner, log_root_path, train_cfg, target_key='dec_0', source_key='dec_0', update_primitive=True)


    runner.env.env_origins = torch.zeros((runner.env.num_envs, 3), device=runner.env.device)
    runner.env.reset()
    runner.env.reset_idx([0], 0)
    runner.env.reset_idx([1], 1)
    obs, critic_obs = runner.env.compute_observations()

    r1_state, r2_state = get_robot_state(runner.env)
    trajectory = [[r1_state],[r2_state]]



    for i in range(env.max_episode_length):
        # ideal
        # runner.env.episode_length_buf+=1
        # runner.env.update_cur_ref()
        # reset_qpos = env.rob_default_qpos.clone().to(device=env.device)
        # reset_qpos[:,0:2] = runner.env.cur_ref_pos_r1[:,0:2]
        # reset_qpos[:,3:7] = runner.env.cur_ref_root_ori_r1
        # runner.env.robot.set_qpos(reset_qpos,  zero_velocity=True )
        # runner.env.scene.step()

        for i in range(2):
            r_state = get_robot_state(runner.env)[i]
            trajectory[i].append(r_state)


        # fact
        actions = runner.alg.act(obs, critic_obs) 
        obs, critic_obs, rewards, dones, infos = runner.env.step(actions)
        obs = runner.obs_normalizer(obs)
        critic_obs = runner.critic_obs_normalizer(critic_obs)




    np.save(log_root_path+'/trained_motion_traj.npy', trajectory)
    



if __name__ == "__main__":
    args = helpers.get_args()
    test(args)

    

