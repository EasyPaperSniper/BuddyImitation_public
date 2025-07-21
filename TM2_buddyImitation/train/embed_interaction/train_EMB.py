
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter

from transmimicV2_interaction import TM2_ROOT_DIR
from transmimicV2_interaction.runner.TM2_Runners import Embeding_Trainer
from transmimicV2_interaction.runner.learning_module import Graph_Embedding, MotionVAE
from transmimicV2_interaction.train.embed_interaction.config import EmbeddingLearningCfg
from transmimicV2_interaction.train.embed_interaction.train_MVAE import convert_st_to_pose

from transmimicV2_interaction.utils.tm_helpers import set_seed
from transmimicV2_interaction.utils.process_intergen.gen_dataset import Motion_dataset
from transmimicV2_interaction.utils.quaterion import *
from transmimicV2_interaction.utils.ratation_conversion import *
from transmimicV2_interaction.utils.process_intergen.interGen_param import *


def train_embedding():
   
    train_cfg = EmbeddingLearningCfg().to_dict()
    set_seed(train_cfg['seed'])
    train_cfg["log_dir"] = TM2_ROOT_DIR + '/results/embedding/'
    
    
    motion_data = Motion_dataset(device=train_cfg['device'])
    motion_data.load_and_form(train_cfg["motion_list"])
    
    trainer = Embeding_Trainer(train_cfg, dataset = motion_data, device=train_cfg['device'])
    print('********************Training Embeded Graph ********************')
    trainer.train(num_learning_iterations= train_cfg['runner']['max_iterations'])


def test_embedding():
    train_cfg = EmbeddingLearningCfg().to_dict()
    set_seed(train_cfg['seed'])
    train_cfg["log_dir"] = TM2_ROOT_DIR + '/results/embedding/'
    EMB_dir = train_cfg["log_dir"] +'/emb_1000_1.pt'
    
    motion_data = Motion_dataset(device=train_cfg['device'])
    motion_data.load_and_form(train_cfg["motion_list"])
    
    embedding_net = Graph_Embedding(
            motion_data.obs_dim,
            motion_data.ig_dim,
            predefine_graph= None,
            **train_cfg['network']).to(device = train_cfg['device'])
        
    MotionVAE_net = MotionVAE(
        motion_data.obs_dim,
        motion_data.obs_dim,
        **train_cfg['network']).to(device = train_cfg['device'])
        

    MotionVAE_net.load(train_cfg['runner']['saved_mvae_dir'], device=train_cfg['device'])
    embedding_net.load(EMB_dir, device=train_cfg['device'])
    
    ############ test each demo motion #############
    saved_traj = {}
    # loss = nn.MSELoss()
    # for motion_name in motion_data.data.keys():
    #     # test motion status
    #     c1_st, c1_stp1, c2_st, c2_stp1, ig = motion_data.sample(batch_size=500, motion_name=motion_name, two_chars=True)
                
    #     windows = c1_st.shape[1]
    #     c1_st_cur, c2_st_cur = c1_st[:,0,:], c2_st[:,0,:]
    #     ig_cur = ig[:,0,:]
        
    #     qk,  qk_hard, att_out, c1_lat, c2_lat = embedding_net.predict(c1_st_cur, c2_st_cur, ig_cur)

    #     c1_pred_stp1 = [MotionVAE_net.decode(c1_st_cur, c1_lat)]
    #     c2_pred_stp1 = [MotionVAE_net.decode(c2_st_cur, c2_lat)]

        
    #     for i in range(1, windows):
    #         c1_st_cur, c2_st_cur = c1_pred_stp1[-1], c2_pred_stp1[-1]

    #         ig_cur = ig[:,i,:]
    #         qk,  qk_hard, att_out, c1_lat, c2_lat =embedding_net.predict(c1_st_cur, c2_st_cur, ig_cur)

    #         c1_pred_stp1.append(MotionVAE_net.decode(c1_st_cur, c1_lat))
    #         c2_pred_stp1.append(MotionVAE_net.decode(c2_st_cur, c2_lat))

        
    #     c1_pred_stp1 = torch.cat([x[:,None] for x in c1_pred_stp1], dim=1)
    #     c2_pred_stp1 = torch.cat([x[:,None] for x in c2_pred_stp1], dim=1)
                                 
    #     reconst_loss = (loss(c1_stp1, c1_pred_stp1) + loss(c2_stp1, c2_pred_stp1)) 
    #     print(motion_name, reconst_loss.item())
        
    ##### test motion traj
    motion_name = 'sparring'
    cur_motion = motion_data.data[motion_name]
    print(cur_motion['clip_SE_idx'])
    saved_traj[motion_name+'_c1'] = []
    saved_traj[motion_name+'_c2']= []
    saved_traj[motion_name+'_gt_c1']= []
    saved_traj[motion_name+'_gt_c2']= []
    saved_traj[motion_name+'_emb']= []
    
    
    motion_length = min(cur_motion['clip_SE_idx'][1] - cur_motion['clip_SE_idx'][0]-1,500)

    with torch.inference_mode():
        for i in range(motion_length-1):
            step_dix = i
            c1_gt_st =  motion_data.data[motion_name]['c1_st'][step_dix].unsqueeze(0)
            c2_gt_st =  motion_data.data[motion_name]['c2_st'][step_dix].unsqueeze(0)
            c1_gt_stp1 =  motion_data.data[motion_name]['c1_st'][step_dix+1].unsqueeze(0)
            c2_gt_stp1 =  motion_data.data[motion_name]['c2_st'][step_dix+1].unsqueeze(0)
            ig_t = motion_data.data[motion_name]['IG'][step_dix].unsqueeze(0)
            

   
            qk,  qk_hard, att_out, c1_lat, c2_lat = embedding_net.predict(c1_gt_st, c2_gt_st, ig_t)
            c1_pred_stp1 = MotionVAE_net.decode(c1_gt_st, c1_lat)
            c2_pred_stp1 = MotionVAE_net.decode(c2_gt_st, c2_lat)
            
            # print(qk_hard.shape)

            
            saved_traj[motion_name+'_c1'].append(convert_st_to_pose(c1_pred_stp1))
            saved_traj[motion_name+'_c2'].append(convert_st_to_pose(c2_pred_stp1))
            saved_traj[motion_name+'_gt_c1'].append(convert_st_to_pose(c1_gt_stp1))
            saved_traj[motion_name+'_gt_c2'].append(convert_st_to_pose(c2_gt_stp1))
            saved_traj[motion_name+'_emb'].append(np.sum(qk_hard.cpu().numpy(), axis=(0,1,2)))
            # print(np.sum(qk_hard.cpu().numpy(), axis=(0,1,2)).reshape(22,22))
            
            
        np.save(train_cfg["log_dir"]+'/test_traj/{}/emb_traj'.format(motion_name), saved_traj[motion_name+'_emb'])
        np.save(train_cfg["log_dir"]+'/test_traj/{}/c1_gt_traj'.format(motion_name), saved_traj[motion_name+'_gt_c1'])
        np.save(train_cfg["log_dir"]+'/test_traj/{}/c2_gt_traj'.format(motion_name), saved_traj[motion_name+'_gt_c2'])
        np.save(train_cfg["log_dir"]+'/test_traj/{}/c1_pred_traj'.format(motion_name), saved_traj[motion_name+'_c1'])
        np.save(train_cfg["log_dir"]+'/test_traj/{}/c2_pred_traj'.format(motion_name), saved_traj[motion_name+'_c2'])
        
    plot_2d_AIG(saved_traj[motion_name+'_emb'], save_dir=train_cfg["log_dir"], motion_name=motion_name)
        
        
    
        
        
        
def heatmap(data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="",**kwargs):
    if ax is None:
        ax = plt.gca()

    # if cbar_kw is None:
    #     cbar_kw = {saved_traj[motion_name+'_emb']}


    im = ax.imshow(data,  **kwargs)
    # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

     # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",rotation_mode="anchor")
    
    return ax, im




def plot_2d_AIG(data, save_dir, motion_name='sparring'):
    data =np.array(data)
    print(data.shape)
    data = data.reshape(data.shape[0], 1, 22,22)
    motion_length = data.shape[0]
    metadata = dict(title='Viz_AIG',artist='EPS')
    writer = FFMpegFileWriter(fps=30, metadata=metadata)

    row_labels = ['c2_{}'.format(i) for i in spml_joint_official_name]
    col_labels = ['c1_{}'.format(i) for i in spml_joint_official_name]

    fig = plt.figure(figsize=(12,9))
    ax =  fig.add_subplot(111)
    ax.set_title('AIG_{}'.format(motion_name))
    pic_setting = {
        'vmin': 0,
        'vmax': 1.1,
    }
    ax, im = heatmap(np.sum(data[0], axis=0), row_labels, col_labels, ax=ax, cbarlabel='attention', **pic_setting)

    with writer.saving(fig,save_dir+'/test_traj/{}/viz_emb.mp4'.format(motion_name), 100):
        for i in range(100):
            ax.imshow(np.sum(data[i], axis=0), **pic_setting)
            writer.grab_frame()

            
        

if __name__ == '__main__':
    train_embedding()
    test_embedding()
