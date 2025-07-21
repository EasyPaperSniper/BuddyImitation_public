

import os
import traceback
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from transmimicV2_interaction import TM2_ROOT_DIR
from transmimicV2_interaction.utils.process_intergen.gen_dataset import Motion_dataset
from transmimicV2_interaction.utils.quaterion import *
from transmimicV2_interaction.utils.ratation_conversion import *  
from transmimicV2_interaction.utils.kinematics_rollout import  KIN_SOLVERS
from transmimicV2_interaction.utils.helpers import class_to_dict, set_seed  
from transmimicV2_interaction.runner.learning_module import  Graph_Embedding, MotionVAE
from transmimicV2_interaction.train.embed_interaction.config import MotionVAELearningCfg,EmbeddingLearningCfg



kin_solver = KIN_SOLVERS['human']

def simulate(c1_pose,c2_pose, c1_nxt_st,  c2_nxt_st, data, step, viz_GT=True):
    c1_root_pos = c1_pose[:,0:7].clone()
    c2_root_pos = c2_pose[:,0:7].clone()


    c1_root, c1_joint = kin_solver.process_output_data(c1_root_pos, c1_nxt_st)
    c2_root, c2_joint = kin_solver.process_output_data(c2_root_pos, c2_nxt_st)    
    
    
    c1_ori_root = data['c1_ori_motion'][step, 66:66+7]
    c2_ori_root = data['c2_ori_motion'][step, 66:66+7]
    c1_ori_joint = data['c1_ori_motion'][step, 79:79+63]
    c2_ori_joint = data['c2_ori_motion'][step, 79:79+63]
    
    if viz_GT:
        c1_root[-1] = c1_ori_root
        c2_root[-1] = c2_ori_root
        c1_joint[-1] = c1_ori_joint
        c2_joint[-1] = c2_ori_joint
    
    return torch.cat([c1_root, c1_joint], dim=-1), torch.cat([c2_root, c2_joint], dim=-1)
    


def set_to_init( data, step):
    c1_root = data['c1_ori_motion'][step, 66:66+7].unsqueeze(0).repeat(2, 1)
    c2_root = data['c2_ori_motion'][step, 66:66+7].unsqueeze(0).repeat(2, 1)
    c1_joint = data['c1_ori_motion'][step, 79:79+63].unsqueeze(0).repeat(2, 1)
    c2_joint = data['c2_ori_motion'][step, 79:79+63].unsqueeze(0).repeat(2, 1)
    
    c1_pose = torch.cat([c1_root, c1_joint], dim=-1)
    c2_pose = torch.cat([c2_root, c2_joint], dim=-1)
    return c1_pose, c2_pose


def test_dataset(dataset, motion_name='sparring'):
    motion_length = dataset.data_norm[motion_name]['c1_st'].shape[0]
    generated_motion = {'c1': [],
                        'c2': []}

    for step in range(motion_length):
        if step in dataset.data[motion_name]['clip_SE_idx']:
            c1_pose, c2_pose = set_to_init( dataset.data[motion_name], step)
            generated_motion['c1'].append(c1_pose)
            generated_motion['c2'].append(c2_pose)

        c1_nxt_st = dataset.data_norm[motion_name]['c1_nxt_st'][step] * dataset.nxt_st_std + dataset.nxt_st_mean     
        c2_nxt_st = dataset.data_norm[motion_name]['c2_nxt_st'][step] * dataset.nxt_st_std + dataset.nxt_st_mean
        
        c1_nxt_st = c1_nxt_st.unsqueeze(0).repeat(2, 1)
        c2_nxt_st = c2_nxt_st.unsqueeze(0).repeat(2, 1)

        c1_pose, c2_pose = simulate(c1_pose, c2_pose, c1_nxt_st,  c2_nxt_st, dataset.data[motion_name], step)

        generated_motion['c1'].append(c1_pose)
        generated_motion['c2'].append(c2_pose)
        step+=1 

    for key, value in generated_motion.items():
        generated_motion[key] =  torch.cat([x[None,:] for x in value], dim=0)

    torch.save(generated_motion,TM2_ROOT_DIR+'/results/saved/trajectories/dataset_test/{}.pt'.format(motion_name))


def test_MAVE_prediction(dataset, MotionVAE_net, motion_name='sparring', windows=10):
    
    motion_length = dataset.data_norm[motion_name]['c1_st'].shape[0]
    generated_motion = {'c1': [],
                        'c2': []}

    for step in range(motion_length):
        if step % windows==0 or step in dataset.data[motion_name]['clip_SE_idx']:
            c1_pose, c2_pose = set_to_init(dataset.data[motion_name], step)
            generated_motion['c1'].append(c1_pose)
            generated_motion['c2'].append(c2_pose)

            c1_st = dataset.data_norm[motion_name]['c1_st'][step].unsqueeze(0)     
            c2_st = dataset.data_norm[motion_name]['c2_st'][step].unsqueeze(0) 
            c1_st = c1_st.repeat(2, 1)
            c2_st = c2_st.repeat(2, 1)  
        
        else:
            c1_st = torch.cat([c1_pred_nxt_st_norm[:,2:3], c1_pred_nxt_st_norm[:,5:]],dim=1)
            c2_st = torch.cat([c2_pred_nxt_st_norm[:,2:3], c2_pred_nxt_st_norm[:,5:]],dim=1)

            
        c1_nxt_st = dataset.data_norm[motion_name]['c1_nxt_st'][step].unsqueeze(0)     
        c2_nxt_st = dataset.data_norm[motion_name]['c2_nxt_st'][step].unsqueeze(0)
        c1_nxt_st = c1_nxt_st.repeat(2, 1)
        c2_nxt_st = c2_nxt_st.repeat(2, 1)
        

        c1_pred_nxt_st_norm,_,_,_ = MotionVAE_net.forward(c1_st, c1_nxt_st)
        c2_pred_nxt_st_norm,_,_,_ = MotionVAE_net.forward(c2_st, c2_nxt_st)




        if not (step % windows==0 or step in dataset.data[motion_name]['clip_SE_idx']):
            c1_pred_nxt_st = c1_pred_nxt_st_norm * dataset.nxt_st_std + dataset.nxt_st_mean
            c2_pred_nxt_st = c2_pred_nxt_st_norm * dataset.nxt_st_std + dataset.nxt_st_mean
            c1_pose, c2_pose =simulate(c1_pose, c2_pose, c1_pred_nxt_st,  c2_pred_nxt_st, dataset.data[motion_name], step)
            generated_motion['c1'].append(c1_pose)
            generated_motion['c2'].append(c2_pose)


        
        step+=1

    for key, value in generated_motion.items():
        generated_motion[key] =  torch.cat([x[None,:] for x in value], dim=0).to(device='cpu')

    torch.save(generated_motion,TM2_ROOT_DIR+'/results/saved/trajectories/MVAE_test/{}.pt'.format(motion_name))


def test_MAVE_random(dataset, MotionVAE_net, motion_name='sparring', windows=200):
    
    motion_length = dataset.data_norm[motion_name]['c1_st'].shape[0]
    generated_motion = {'c1': [],
                        'c2': []}

    for step in range(motion_length):
        if step % windows==0 or step in dataset.data[motion_name]['clip_SE_idx']:
            c1_pose, c2_pose = set_to_init(dataset.data[motion_name], step)
            generated_motion['c1'].append(c1_pose)
            generated_motion['c2'].append(c2_pose)

            c1_st = dataset.data_norm[motion_name]['c1_st'][step].unsqueeze(0)     
            c2_st = dataset.data_norm[motion_name]['c2_st'][step].unsqueeze(0) 
            c1_st = c1_st.repeat(2, 1)
            c2_st = c2_st.repeat(2, 1)  
        
        else:
            c1_st = torch.cat([c1_pred_nxt_st_norm[:,2:3], c1_pred_nxt_st_norm[:,5:]],dim=1)
            c2_st = torch.cat([c2_pred_nxt_st_norm[:,2:3], c2_pred_nxt_st_norm[:,5:]],dim=1)
            
        c1_nxt_st = dataset.data_norm[motion_name]['c1_nxt_st'][step].unsqueeze(0)     
        c2_nxt_st = dataset.data_norm[motion_name]['c2_nxt_st'][step].unsqueeze(0)
        c1_nxt_st = c1_nxt_st.repeat(2, 1)
        c2_nxt_st = c2_nxt_st.repeat(2, 1)
        

        c1_lat_vec = MotionVAE_net.encode(c1_st, c1_nxt_st)
        c1_pred_nxt_st_norm = MotionVAE_net.decode(c1_st, c1_lat_vec)
        c2_lat_vec = MotionVAE_net.encode(c2_st, c2_nxt_st)
        c2_pred_nxt_st_norm = MotionVAE_net.decode(c2_st, c2_lat_vec)

        if not (step % windows==0 or step in dataset.data[motion_name]['clip_SE_idx']):
            c1_pred_nxt_st = c1_pred_nxt_st_norm * dataset.nxt_st_std + dataset.nxt_st_mean
            c2_pred_nxt_st = c2_pred_nxt_st_norm * dataset.nxt_st_std + dataset.nxt_st_mean
            c1_pose, c2_pose =simulate(c1_pose, c2_pose, c1_pred_nxt_st,  c2_pred_nxt_st, dataset.data[motion_name], step, False)
            generated_motion['c1'].append(c1_pose)
            generated_motion['c2'].append(c2_pose)
            
            
        
        step+=1

    for key, value in generated_motion.items():
        generated_motion[key] =  torch.cat([x[None,:] for x in value], dim=0).to(device='cpu')

    torch.save(generated_motion,TM2_ROOT_DIR+'/results/saved/trajectories/MVAE_test/random.pt')


def test_sig(dataset, MotionVAE_net, embeded_net, motion_name='sparring', windows=1):
    motion_length = dataset.data_norm[motion_name]['c1_st'].shape[0]
    motion_length = dataset.data[motion_name]['clip_SE_idx'][1]
    generated_motion = {
                        # 'c1': [],
                        # 'c2': [],
                        'c1_qk': [],
                        'c2_qk': [],
                        'c1_qk_hard': [],
                        'c2_qk_hard': []}

    for step in range(motion_length):
        if step % windows==0 or step in dataset.data[motion_name]['clip_SE_idx']:
            c1_pose, c2_pose = set_to_init(dataset.data[motion_name], step)
            # generated_motion['c1'].append(c1_pose)
            # generated_motion['c2'].append(c2_pose)

            c1_st = dataset.data_norm[motion_name]['c1_st'][step].unsqueeze(0).repeat(2, 1)     
            c2_st = dataset.data_norm[motion_name]['c2_st'][step].unsqueeze(0).repeat(2, 1)
            c1_ig = dataset.data_norm[motion_name]['IG_in_1'][step].unsqueeze(0).repeat(2, 1,1)
            c2_ig = dataset.data_norm[motion_name]['IG_in_2'][step].unsqueeze(0).repeat(2, 1,1)

        else:
            c1_st = torch.cat([c1_pred_nxt_st_norm[:,2:3], c1_pred_nxt_st_norm[:,5:]],dim=1)
            c2_st = torch.cat([c2_pred_nxt_st_norm[:,2:3], c2_pred_nxt_st_norm[:,5:]],dim=1)

        
        qk_c1,qk_hard_c1,attout_c1,c1_enc_var = embeded_net.predict(c1_st, c1_ig)
        qk_c2,qk_hard_c2,attout_c2,c2_enc_var = embeded_net.predict(c2_st, c2_ig)
        generated_motion['c1_qk'].append(qk_c1.squeeze(-2))
        generated_motion['c2_qk'].append(qk_c2.squeeze(-2))
        # generated_motion['c1_attout'].append(attout_c1)
        # generated_motion['c2_attout'].append(attout_c2)
        generated_motion['c1_qk_hard'].append(qk_hard_c1.squeeze(-2))
        generated_motion['c2_qk_hard'].append(qk_hard_c2.squeeze(-2))

        # print(qk_hard_c1)
        
        c1_pred_nxt_st_norm = MotionVAE_net.decode(c1_st, c1_enc_var)
        c2_pred_nxt_st_norm = MotionVAE_net.decode(c2_st, c2_enc_var)




        if not (step % windows==0 or step in dataset.data[motion_name]['clip_SE_idx']):
            c1_pred_nxt_st = c1_pred_nxt_st_norm * dataset.nxt_st_std + dataset.nxt_st_mean
            c2_pred_nxt_st = c2_pred_nxt_st_norm * dataset.nxt_st_std + dataset.nxt_st_mean
            c1_ig, c2_ig = kin_solver.gen_nxt_ig(c1_pose[:,0:7], c2_pose[:,0:7], c1_ig, c2_ig, 
                                                     c1_pred_nxt_st, c2_pred_nxt_st)
            c1_pose, c2_pose =simulate(c1_pose, c2_pose, c1_pred_nxt_st,  c2_pred_nxt_st, dataset.data[motion_name], step)
            
            # generated_motion['c1'].append(c1_pose)
            # generated_motion['c2'].append(c2_pose)
            
        
        step+=1

    for key, value in generated_motion.items():
        generated_motion[key] =  torch.cat([x[None,:] for x in value], dim=0).to(device='cpu')
       

    torch.save(generated_motion,TM2_ROOT_DIR+'/results/saved/trajectories/embd_test/{}.pt'.format(motion_name))






if __name__ == '__main__':
    device = 'cuda:0'
    dataset = Motion_dataset(device=device)
    dataset.load_and_form(['sparring', 'handshake', 'rock-paper','reach-hug-short', 'hold-hand-and-circle'])
    
    MVAE_dir =  TM2_ROOT_DIR + '/results/saved/models/mvae_030500_8_256_256.pt'
    train_cfg = class_to_dict(EmbeddingLearningCfg())
    MotionVAE_net = MotionVAE(dataset.obs_dim, 
                            dataset.nxt_obs_dim,
                                **train_cfg['network']).to(device=device)
    MotionVAE_net.load(MVAE_dir, device=device)  
    
    
    EMBED_dir =  TM2_ROOT_DIR + '/results/embedding/emb_sparring_1000_1.pt'
    embedding_net = Graph_Embedding(
        dataset.obs_dim, 
        dataset.ig_dim,
        **train_cfg['network']).to(device=device)
    embedding_net.load(EMBED_dir, device=device)
     
    # test_dataset(dataset)
    # for motion_name in ['sparring', 'handshake', 'rock-paper','reach-hug-short', 'hold-hand-and-circle']:
    #     test_MAVE_prediction( dataset, MotionVAE_net, motion_name=motion_name)
    # test_MAVE_random( dataset, MotionVAE_net, motion_name='sparring')
        
    test_sig(dataset, MotionVAE_net, embedding_net)
    
 