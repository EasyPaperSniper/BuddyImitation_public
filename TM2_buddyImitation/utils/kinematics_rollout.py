import torch
import numpy as np
import matplotlib.pyplot as plt

from transmimicV2_interaction import TM2_ROOT_DIR, TM2_ENVS_DIR
from transmimicV2_interaction.utils.quaterion import *
from transmimicV2_interaction.utils.ratation_conversion import *  
import transmimicV2_interaction.utils.pytorch_kinematics as pk
from transmimicV2_interaction.utils.process_intergen.interGen_param import *


class kinematics_solver:
    def __init__(self,
                 urdf_path = TM2_ENVS_DIR + '/assets/humanoid/smpl_humanoid/smpl_humanoid_loose.urdf') -> None:
        self.urdf_file = open(urdf_path).read()
        self.chain = pk.build_chain_from_urdf(self.urdf_file)
    
    
    def gen_nxt_ig(self,cur_root_1, cur_root_2, cur_ig_1, cur_ig_2, nxt_st_1, nxt_st_2):
        
        nxt_root_1, nxt_joint_1 = self.process_output_data(cur_root_1, nxt_st_1)
        nxt_root_2, nxt_joint_2 = self.process_output_data(cur_root_2, nxt_st_2)

        nxt_root_heading_quat_1, _, _ = self.get_headings(nxt_root_1[:, 3:7])
        nxt_root_heading_quat_2, _,_ = self.get_headings(nxt_root_2[:, 3:7])
        
        keypoint_pos_c1_w = self.get_keypoint_pos_w(nxt_root_1, nxt_joint_1)
        keypoint_pos_c2_w = self.get_keypoint_pos_w(nxt_root_2, nxt_joint_2)
        
        IG_pos_in_1 = keypoint_pos_c1_w.unsqueeze(2) - keypoint_pos_c2_w.unsqueeze(1)
        IG_pos_in_1 = IG_pos_in_1.reshape(IG_pos_in_1.shape[0], -1, 3)
        
        IG_pos_in_2 = keypoint_pos_c2_w.unsqueeze(2) - keypoint_pos_c1_w.unsqueeze(1)
        IG_pos_in_2 = IG_pos_in_2.reshape(IG_pos_in_2.shape[0], -1, 3)
        
        IG_in_1 = qrot(qinv(nxt_root_heading_quat_1).unsqueeze(1).repeat(1, IG_pos_in_1.shape[1], 1), IG_pos_in_1)
        IG_in_2 = qrot(qinv(nxt_root_heading_quat_2).unsqueeze(1).repeat(1, IG_pos_in_2.shape[1], 1), IG_pos_in_2)
        
        IG_in_1_vel = IG_in_1 - cur_ig_1[:, :, 0:3]
        IG_in_2_vel = IG_in_2 - cur_ig_2[:, :, 0:3]
        
        IG_in_1 = torch.cat([IG_in_1[:-1], IG_in_1_vel], dim=-1)
        IG_in_2 = torch.cat([IG_in_2[:-1], IG_in_2_vel], dim=-1)
            
        return IG_in_1, IG_in_2

        


    def get_keypoint_pos_w(self,root_state, joint_pos):
        self.chain.to(device=root_state.device)

        root_pos, root_quat = root_state[:, 0:3], root_state[:, 3:7]
        
        joint_pos_rearranged = (joint_pos[:, Smpl2Ig_kinematic_tree]).reshape(joint_pos.shape[0], 21,3)[:, Ig2PyKin_tree]
        joint_pos_r = torch.zeros_like(joint_pos_rearranged,device=root_pos.device)
        joint_pos_w = torch.zeros_like(joint_pos_rearranged,device=root_pos.device)
        
        
        chain_angle = joint_pos_rearranged.reshape(joint_pos_rearranged.shape[0], -1)
        tg_batch = self.chain.forward_kinematics(chain_angle)   
        
        for j in range(len(spml_JL_name)):
                joint_pos_r[:,j] = tg_batch[spml_JL_name[j]].get_matrix()[:, :3, 3]
        root_quat = root_quat.unsqueeze(1).repeat(1, joint_pos_r.shape[1], 1)
        joint_pos_w = qrot(root_quat, joint_pos_r) + root_pos.unsqueeze(1)
        keypoint_pos = torch.cat([root_pos.unsqueeze(1), joint_pos_w], dim=1)
            

        return keypoint_pos


    def get_headings(self, root_quat):
        forward = torch.tensor([1., 0, 0], device=root_quat.device).unsqueeze(0).repeat(root_quat.shape[0], 1)
        rot_dir = qrot(root_quat, forward)
        rot_heading = torch.atan2(rot_dir[:, 1], rot_dir[:, 0])
        root_heading_w = torch.zeros((root_quat.shape[0],3), device=root_quat.device)
        root_heading_w[:, 0] = torch.cos(rot_heading)
        root_heading_w[:, 1] = torch.sin(rot_heading)
        root_heading_quat = qbetween(forward, root_heading_w)
        return root_heading_quat, rot_heading, forward


    def process_output_data(self, root_pos, nxt_st):
        heading_quat, heading_angle, forward = self.get_headings(root_pos[:, 3:7])
        delta_pos = qrot(heading_quat, nxt_st[:, 0:3])
        root_pos[:, 0:2] += delta_pos[:,0:2]
        root_pos[:, 2] = nxt_st[:, 2]
        
        root_heading_w = torch.zeros((root_pos.shape[0],3), device=root_pos.device)
        root_heading_w[:,0:2] = nxt_st[:, 3:5]
        nxt_heading_quat = qmul( heading_quat, qbetween(forward,root_heading_w))
        nxt_root_quat_r = matrix_to_quaternion(cont6d_to_matrix(nxt_st[:,5:11]))
        nxt_root_quat = qmul(nxt_root_quat_r, nxt_heading_quat)

        joint_angle = nxt_st[:, 17:17+63]
        
        return torch.cat([root_pos[:,0:3], nxt_root_quat], dim=1), joint_angle
    

KIN_SOLVERS ={
    'human':kinematics_solver(),
    # 'child':kinematics_solver(urdf_path =TM2_ENVS_DIR + '/assets/humanoid/child/child.urdf'),
    # 'Go2Ar':kinematics_solver(urdf_path =TM2_ENVS_DIR + '/assets/Go2Arx5/urdf/go2_arx5_finray_no_damping.urdf')
}