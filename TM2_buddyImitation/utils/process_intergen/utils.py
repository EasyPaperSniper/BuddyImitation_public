

import numpy as np
import torch
from scipy.signal import savgol_filter

from TM2_buddyImitation import TM2_ROOT_DIR, TM2_ENVS_DIR
from TM2_buddyImitation.utils.process_intergen.interGen_param import *
from TM2_buddyImitation.utils.quaterion import *
from TM2_buddyImitation.utils.ratation_conversion import *



trans_matrix = torch.Tensor([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]])


max_length = 300
min_length = 15




def process_interGen_data(motion_index):

    file_path1 = TM2_ROOT_DIR+'/motion_data/interGen/ori/{}/{}.npy'.format('person1', str(motion_index))
    file_path2 = TM2_ROOT_DIR+'/motion_data/interGen/ori/{}/{}.npy'.format('person2', str(motion_index))
    motion1 = load_motion(file_path1, min_length=min_length)
    motion2 = load_motion(file_path2, min_length=min_length)



    motion1, root_quat_init1, root_pos_init1 = process_motion_tm(motion1, 0.001, 0, n_joints=22)
    motion2, root_quat_init2, root_pos_init2 = process_motion_tm(motion2, 0.001, 0, n_joints=22)


    r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
    angle = np.arctan2(r_relative[:, 3:4], r_relative[:, 0:1])


    xy = qrot_np(qinv_np(root_quat_init1), root_pos_init2 - root_pos_init1)[:, [0, 1]]
    relative = np.concatenate([angle, xy], axis=-1)[0]
    motion2 = rigid_transform(relative, motion2)

    np.save(TM2_ROOT_DIR+'/motion_data/interGen/processed/{}/{}.npy'.format('c1', str(motion_index)), motion1)
    np.save(TM2_ROOT_DIR+'/motion_data/interGen/processed/{}/{}.npy'.format('c2', str(motion_index)), motion2)

    return motion1, motion2, r_relative[0]




def load_motion(file_path, min_length):
    try:
        motion = np.load(file_path).astype(np.float32)
    except:
        print("error: ", file_path)
        return None, None
    joint_position = motion[:, :22 * 3]


    if motion.shape[0] < min_length:
        return None, None

    return joint_position



def rigid_transform(relative, data):

    global_positions = data[..., :22 * 3].reshape(data.shape[:-1] + (22, 3))
    root_vel = data[..., 73: 73+3]
    root_quat = data[..., 66+3:66+7]
    root_ang_vel = data[..., 76:79]

    relative_rot = relative[0]
    relative_t = relative[1:3]
    relative_r_rot_quat = np.zeros(global_positions.shape[:-1] + (4,))
    relative_r_rot_quat[..., 0] = np.cos(relative_rot)
    relative_r_rot_quat[..., 3] = np.sin(relative_rot)
    global_positions = qrot_np(relative_r_rot_quat, global_positions)
    global_positions[..., [0, 1]] += relative_t
    data[..., :22 * 3] = global_positions.reshape(data.shape[:-1] + (-1,))


    relative_r_rot_quat = np.zeros(global_positions.shape[:-2] + (4,))
    relative_r_rot_quat[..., 0] = np.cos(relative_rot)
    relative_r_rot_quat[..., 3] = np.sin(relative_rot)
    data[..., 73: 73+3] = qrot_np(relative_r_rot_quat, root_vel)
    data[..., 76: 79] = qrot_np(relative_r_rot_quat, root_ang_vel)
    data[..., 66+3:66+7] = qmul_np(relative_r_rot_quat,root_quat )
    data[..., 66:66+3] = data[..., :3]

    return data




def process_motion_tm(motion, feet_thre, prev_frames, n_joints):

    positions = motion[:, :n_joints*3].reshape(-1, n_joints, 3)
    positions = savgol_filter(positions, 5, 2, axis=0)  # smooth the data


    positions = np.einsum("mn, tjn->tjm", trans_matrix, positions)

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[2]
    positions[:, :, 2] -= floor_height


    '''XZ at origin'''
    root_pos_init = positions[prev_frames]
    root_pose_init_xy = root_pos_init[0] * np.array([1, 1, 0])
    positions = positions - root_pose_init_xy

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across = root_pos_init[r_hip] - root_pos_init[l_hip]
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around z-axis
    forward_init = np.cross(np.array([[0, 0, 1]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[1, 0, 0]])
    root_quat_init = qbetween_np(target , forward_init)
    root_quat_init_for_all = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions = qrot_np(qinv_np(root_quat_init_for_all), positions)

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([0.12, 0.05])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l_h = positions[:-1,fid_l,2]
        feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r_h = positions[:-1,fid_r,2]
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)



    ''' Get Root Facing Direction and root ori'''
    across = positions[:, l_hip] - positions[:,r_hip]
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
    forward = np.cross(across,np.array([[0, 0, 1]]), axis=-1)
    forward = forward/ np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[1, 0, 0]])
    root_facing = qbetween_np(target,forward )


    # print(across, across_2)
    across_2 = positions[:, 0] - positions[:,r_hip]
    across_2 = across_2 / np.sqrt((across_2 ** 2).sum(axis=-1))[..., np.newaxis]
    
    root_point = np.cross(across, across_2, axis=-1)
    root_point = root_point / np.sqrt((root_point ** 2).sum(axis=-1)+ 1e-5)[..., np.newaxis]

    root_ori = qmul_np(root_facing, qbetween_np(forward,root_point))
    root_ori_xyz = matrix_to_axis_angle(torch.tensor(quaternion_to_matrix_np(root_ori))).detach().cpu().numpy()  

    


    '''Get Joint Rotation Representation'''
    # some simplification here, only use root facing direction
    joint_rot_data = calc_joint_angle_via_IK(root_ori, positions)
    joint_rot_data_matrix = torch.tensor(quaternion_to_matrix_np(joint_rot_data))
    joint_rot_data_euler = (matrix_to_euler_angles(joint_rot_data_matrix,'XYZ').reshape(-1,21*3))[:,Ig2Smpl_kinematic_tree]
    joint_rot_data_euler = joint_rot_data_euler.detach().cpu().numpy()  
    joint_rot_data_euler = savgol_filter(joint_rot_data_euler, 5, 2, axis=0)  # smooth the data



    '''Get Joint Rotation Invariant Position Represention'''
    joint_positions = positions.reshape(len(positions), -1)
    
    root_positions = joint_positions[:, :3]
    root_vels = root_positions[1:] - root_positions[:-1]
    root_vels = root_vels.reshape(len(root_vels), -1)
    root_ang_vels = root_ori_xyz[1:] - root_ori_xyz[:-1]
    root_ang_vels = root_ang_vels.reshape(len(root_ang_vels), -1)



    joint_rot_vels = joint_rot_data_euler[1:] - joint_rot_data_euler[:-1]
    joint_rot_vels = joint_rot_vels.reshape(len(joint_rot_vels), -1)

    data = joint_positions[:-1] # dims = 66
    data = np.concatenate([data, root_positions[:-1]], axis=-1) # dims = 3
    data = np.concatenate([data, root_ori[:-1]], axis=-1) # dims = 4
    data = np.concatenate([data, root_vels], axis=-1)  # dims = 3
    data = np.concatenate([data, root_ang_vels], axis=-1)  # dims = 3
    data = np.concatenate([data, joint_rot_data_euler[:-1]], axis=-1)  # dims = 63
    data = np.concatenate([data, joint_rot_vels], axis=-1) # dims = 63
    data = np.concatenate([data, feet_l, feet_r], axis=-1) # dims = 4


    return data, root_quat_init, root_pose_init_xy[None]





def calc_joint_angle_via_IK(root_ori, positions):
    joint_ori_global = np.ones(positions.shape[:-2]+( 22, 4))*np.array([1,0,0,0])
    joint_ori_global[:,0] = root_ori
    joint_angle = np.ones(positions.shape[:-2]+( 21, 4))*np.array([1,0,0,0])

    
    for chain in joint_kinematics_tree:
        for i in range(1,len(chain)-1):
            if chain[i] in [12,13,14,15]:
                quat = np.ones_like(root_ori)*np.array([1,0,0,0])
            else:
                delta_pos = positions[:,chain[i+1]] - positions[:,chain[i]]
                delta_local = qrot_np(qinv_np(joint_ori_global[:,chain[i-1]]),delta_pos)
                delta_local = delta_local / np.sqrt((delta_local ** 2).sum(axis=-1))[..., np.newaxis]
                quat = qbetween_np( np.ones_like( delta_local)*offset_vec[chain[i]],delta_local)
            

            joint_angle[:,chain[i]-1] = quat
            joint_ori_global[:,chain[i]] = qmul_np(joint_ori_global[:,chain[i-1]], quat)
            


    return joint_angle





if __name__ =='__main__':
    for index in [5100]:
        process_interGen_data(index)
