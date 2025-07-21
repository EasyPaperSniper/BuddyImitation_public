from transmimicV2_interaction import TM2_ROOT_DIR
from transmimicV2_interaction.runner.TM2_Runners import MotionVAE_Trainer
from transmimicV2_interaction.train.embed_interaction.config import MotionVAELearningCfg
from transmimicV2_interaction.runner.learning_module import MotionVAE

from transmimicV2_interaction.utils.tm_helpers import set_seed
from transmimicV2_interaction.utils.process_intergen.gen_dataset import Motion_dataset
from transmimicV2_interaction.utils.quaterion import *
from transmimicV2_interaction.utils.ratation_conversion import *


def train_MVAE():
    train_cfg = MotionVAELearningCfg().to_dict()
    set_seed(train_cfg['seed'])
    train_cfg["log_dir"] = TM2_ROOT_DIR + '/results/MVAE'
    
    motion_data = Motion_dataset(device=train_cfg['device'])
    motion_data.load_and_form(train_cfg["motion_list"])
    
    trainer = MotionVAE_Trainer(train_cfg, dataset = motion_data, device=train_cfg['device'])
    print('********************Training MotionVAE********************')
    trainer.train(num_learning_iterations= train_cfg['runner']['max_iterations'])
    

def convert_st_to_pose(st):
    root_pos = st[:,:3]
    root_quat = matrix_to_quaternion(cont6d_to_matrix(st[:,3:9]))
    joint_pose = st[:,12:]
    return torch.cat([root_pos, root_quat, joint_pose], dim=1).cpu().numpy()


def test_MVAE():
    train_cfg = MotionVAELearningCfg().to_dict()
    set_seed(train_cfg['seed'])
    train_cfg["log_dir"] = TM2_ROOT_DIR + '/results/MVAE'
    MVAE_dir = train_cfg["log_dir"] +'/mvae_1000_1.pt'
    
    motion_data = Motion_dataset(device=train_cfg['device'])
    motion_data.load_and_form(train_cfg["motion_list"])
    
    MotionVAE_net = MotionVAE(motion_data.obs_dim, 
                             motion_data.obs_dim,
                                **train_cfg['network']).to(device=train_cfg['device'])
    MotionVAE_net.load(MVAE_dir, device=train_cfg['device'])  
    
    
    ############ test each demo motion #############
    saved_traj = {}
    for motion_name in motion_data.data.keys():
        cur_motion = motion_data.data[motion_name]
        motion_length = min(cur_motion['clip_SE_idx'][1] - cur_motion['clip_SE_idx'][0]-1,600)

        with torch.inference_mode():
            for i in range(motion_length-1):
                gt_st =  motion_data.data[motion_name]['c1_st'][i].unsqueeze(0)
                gt_stp1 = motion_data.data[motion_name]['c1_st'][i+1].unsqueeze(0)
                
                if i % 30 == 0:
                    cur_st = gt_st.clone()
                    if i ==0:
                        saved_traj[motion_name+'_mvae'] = [convert_st_to_pose(cur_st)]
                        saved_traj[motion_name+'_gt'] = [convert_st_to_pose(gt_st)]
                else:
                    cur_st = pred_stp1
                lat_var = MotionVAE_net.encode(cur_st, gt_stp1)
                pred_stp1 = MotionVAE_net.decode(cur_st, lat_var)
                
                saved_traj[motion_name+'_mvae'].append(convert_st_to_pose(pred_stp1))
                saved_traj[motion_name+'_gt'].append(convert_st_to_pose(gt_st))
                
                
            np.save(train_cfg["log_dir"]+'/test_traj/mvae_traj_'+motion_name, saved_traj[motion_name+'_mvae'])
            np.save(train_cfg["log_dir"]+'/test_traj/gt_traj_'+motion_name, saved_traj[motion_name+'_gt'])
            
        
     

if __name__ == '__main__':
    # train_MVAE()
    test_MVAE()
