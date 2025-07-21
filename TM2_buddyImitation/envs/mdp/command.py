import torch
from TM2_buddyImitation.utils.math import quat_rotate_inverse, yaw_quat, axis_angle_from_quat, quat_rotate_inverse

class BaseCommandManager():
    def __init__(self, cmd_cfg):
        self.cmd_dof = cmd_cfg.cmd_dof
        self.cmd_interval_s = cmd_cfg.cmd_interval_s
    

    def sample_commands(self, env ):
        env_ids = (env.episode_length_buf % int( self.cmd_interval_s /env.dt)==0).nonzero(as_tuple=False).flatten()
        norm_commands = torch.rand((env_ids.shape[0], self.cmd_dof), device=env.device, requires_grad=False) - 0.5
        env.commands[env_ids] = norm_commands * torch.tensor([[2, 1, 2]], requires_grad=False, device=env.device)
        return env.commands
        


class TM2SimpleCommandManager(BaseCommandManager):

    def sample_commands(self, env):
        return None



class TM2CommandManager(TM2SimpleCommandManager):

    def sample_commands(self, env):
        return None