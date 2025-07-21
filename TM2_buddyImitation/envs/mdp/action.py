import torch
from TM2_buddyImitation.utils.math import quat_rotate_inverse, quat_rotate

class BaseActionManager():
    def compute(self, actions, env):
        clip_actions = env.cfg.normalization.clip_actions
        env.actions = torch.clip(actions, -clip_actions, clip_actions)
        tgt_pos = env.actions * env.cfg.robot.action_scale + env.rob_default_qpos[:,7:]
        return tgt_pos
    



class TM2SimpleActionManager():
    def compute(self, actions, env):

        clip_actions = env.cfg.normalization.clip_actions
        env.actions = torch.clip(actions, -clip_actions, clip_actions)

        actions = actions[:,0:env.num_actions_r1] * env.cfg.robot.action_scale + env.dof_pos

        return actions
    


class TM2ActionManager():
    def compute(self, actions, env):
        '''
        actions = [action_r1, action_r2]
        '''
        clip_actions = env.cfg.normalization.clip_actions
        env.actions = torch.clip(actions, -clip_actions, clip_actions)

        action_r1 = actions[:,0:env.num_actions_r1] * env.cfg.robot.action_scale + env.dof_pos
        action_r2 = actions[:,env.num_actions_r1:] * env.cfg.robot_2.action_scale + env.dof_pos_r2

        return action_r1, action_r2
    

