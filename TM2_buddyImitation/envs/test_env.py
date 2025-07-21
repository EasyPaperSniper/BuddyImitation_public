import genesis as gs
from TM2_buddyImitation.configs.env_cfg import *
from TM2_buddyImitation.envs.TM2_env import TM2_Env



def rollout(env,):
    env.reset()
    while True:
        action = torch.rand((env.num_envs, env.num_actions), device=env.device)
        env.step(action)
        # env.robot.set_qpos(env.rob_default_qpos, zero_velocity=True)
        # env.robot_2.set_qpos(env.rob_default_qpos_r2, zero_velocity=True)
        env.scene.step()





if __name__ == "__main__":
    render = True
    env_cfg = TM2Cfg()
    env = TM2_Env(env_cfg, render=render)




    rollout(env)

    