import torch
import numpy

from transmimicV2_interaction import TM2_ROOT_DIR, TM2_ENVS_DIR
from transmimicV2_interaction.utils.process_intergen.interGen_param import *
from transmimicV2_interaction.utils.quaterion import *
from transmimicV2_interaction.utils.ratation_conversion import *



class Motion_dataset():
    def __init__(self, 
                 dataset_pth='/motion_data/interGen/labeled/',
                 horizon=[1,2,5,10], 
                 windows=5,
                 device='cpu',
                 **kwargs):
        self.dataset_pth = TM2_ROOT_DIR + dataset_pth
        self.data = {}
        self.norm = {}
        self.device = device
        self.horizon = horizon
        self.windows = windows

        
    
    def sample(self, batch_size:int, motion_name=None, two_chars=False):
        if motion_name is None:
            motion_name = numpy.random.choice(list(self.data.keys()))

        motion_data = self.data[motion_name]
        indices = torch.randint(high=int(self.data[motion_name]['valid_idx'].shape[0]), size=(batch_size,)).to(self.device)
        clip_idx = self.data[motion_name]['valid_idx'][indices].unsqueeze(1) + torch.arange(self.windows).unsqueeze(0).to(self.device)

        
        if not two_chars:
            if np.random.randint(2) % 2 == 0:
                return motion_data['c1_st'][clip_idx], motion_data['c1_st'][clip_idx+1]
            else:
                return motion_data['c2_st'][clip_idx], motion_data['c2_st'][clip_idx+1]
        else:
            return  motion_data['c1_st'][clip_idx], motion_data['c1_st'][clip_idx+1], \
                    motion_data['c2_st'][clip_idx], motion_data['c2_st'][clip_idx+1], motion_data['IG'][clip_idx]
                   
               
    def load_and_form(self, motion_names:list):
        self.load_ori_data(motion_names)
        for motion_name in motion_names:
            clip_idx = self.data[motion_name]['clip_SE_idx']
            self.data[motion_name]['valid_idx'] = torch.cat([torch.arange(clip_idx[i], clip_idx[i+1]-self.windows-1) for i in range(len(clip_idx)-1)]).to(self.device)
                    

    def load_ori_data(self, motion_names:list):
        for motion_name in motion_names:
            self.data[motion_name] = torch.load(self.dataset_pth+motion_name+'.pt')
            for key, value in self.data[motion_name].items():
                if key != 'clip_SE_idx':
                    self.data[motion_name][key] = value.to(self.device)
        self.obs_dim = self.data[motion_name]['c1_st'].shape[1]
        self.ig_dim = self.data[motion_name]['IG'].shape[1:]


    def get_norm(self, motion_names):
        if self.norm.get('st_mean') is not None:
            return 
        
        self.load_ori_data(motion_names)
        summary_st = None
        for _ in range(1000):
            for motion_name in motion_names:
                if summary_st is None:
                    summary_st = self.sample(100, motion_name=motion_name, ori_data=True)[1]
                else:
                    summary_st = torch.cat([summary_st, self.sample(100, motion_name=motion_name, ori_data=True)[1]], dim=0)

        self.st_mean = summary_st.mean(dim=0)
        self.st_std = summary_st.std(dim=0)
        self.st_std = torch.where(self.st_std < 1e-3, torch.ones_like(self.st_std), self.st_std)
        self.ig_mean = torch.zeros(self.ig_dim, device=self.device)
        self.ig_std = torch.ones(self.ig_dim, device=self.device)
        

        self.norm ={
            'st_mean': self.st_mean,
            'st_std': self.st_std,
            'ig_mean': self.ig_mean,
            'ig_std': self.ig_std
        }
        torch.save(self.norm, self.dataset_pth+'norm.pt')
 


    def gen_dataset(self, motion_name:str):
        motion_index_set = interGen_motion_idx[motion_name]
        data = {}
        for idx in motion_index_set:
            c1_motion = np.load(TM2_ROOT_DIR+ '/motion_data/interGen/processed/c1/{}.npy'.format(str(idx)))
            c1_motion = torch.from_numpy(c1_motion).to(self.device)
            c2_motion = np.load(TM2_ROOT_DIR+ '/motion_data/interGen/processed/c2/{}.npy'.format(str(idx)))
            c2_motion = torch.from_numpy(c2_motion).to(self.device)
            
            c1_st = self.gen_st_nxtst(c1_motion)
            c2_st = self.gen_st_nxtst(c2_motion)
            IG = self.gen_IG(c1_motion, c2_motion)
            
            print(c1_st.shape, c2_st.shape, IG.shape,)
            
            if data.get('c1_st') is None:
                data['c1_st'] = c1_st
                data['c2_st'] = c2_st
                data['IG'] = IG
                data['clip_SE_idx'] = [0, c1_st.shape[0]]
                data['c1_ori_motion'] = c1_motion
                data['c2_ori_motion'] = c2_motion
            else:
                data['c1_st'] = torch.cat([data['c1_st'], c1_st], dim=0)
                data['c2_st'] = torch.cat([data['c2_st'], c2_st], dim=0)
                data['IG'] = torch.cat([data['IG'], IG], dim=0)
                data['clip_SE_idx'].append(data['clip_SE_idx'][-1] + c1_st.shape[0])
                data['c1_ori_motion'] = torch.cat([data['c1_ori_motion'], c1_motion], dim=0)
                data['c2_ori_motion'] = torch.cat([data['c2_ori_motion'], c2_motion], dim=0)
        print(data['clip_SE_idx'])
        torch.save(data, self.dataset_pth+motion_name+'.pt')
            
            
    
    def get_headings(self, motion):
        root_quat = motion[:, 66+3:66+7]
        forward = torch.tensor([1., 0, 0], device=root_quat.device).unsqueeze(0).repeat(motion.shape[0], 1)
        rot_dir = qrot(root_quat, forward)
        rot_heading = torch.atan2(rot_dir[:, 1], rot_dir[:, 0])
        root_heading_w = torch.zeros([motion.shape[0], 3], device=root_quat.device)
        root_heading_w[:, 0] = torch.cos(rot_heading)
        root_heading_w[:, 1] = torch.sin(rot_heading)
        root_heading_quat = qbetween(forward, root_heading_w )
        return root_heading_quat, rot_heading


    def gen_st_nxtst(self, motion):
        root_heading_quat, root_heading_angle = self.get_headings(motion)
        root_pos = motion[:, 66:66+3]
        root_quat = motion[:, 66+3:66+7]
        joint_pose = motion[:, 79:79+63]
        
        root_lin_vel = motion[:, 66+7:66+10]
        root_ang_vel = motion[:, 66+10:66+13]
       
        # joint_vel = motion[:, 79+63:79+63*2]
        # root_quat_r = qmul(root_quat, qinv(root_heading_quat))
        # root_lin_vel_r = qrot(qinv(root_heading_quat), root_lin_vel)
        # root_ang_vel_r = qrot(qinv(root_heading_quat), root_ang_vel)
        
        
        # nxt_root_pos = root_pos[1:] - root_pos[:-1]
        # nxt_root_pos[:,2] = 0
        # nxt_root_pos_r = qrot(qinv(root_heading_quat[:-1]), nxt_root_pos)
        # nxt_root_pos_r[:,2] = root_pos[1:,2]
        # nxt_root_heading = root_heading_angle[1:] - root_heading_angle[:-1]
        

        st = torch.cat([root_pos, 
                         quaternion_to_cont6d(root_quat), 
                         root_lin_vel,
                         joint_pose, 
                         ], dim=1)
   
        return st[:-1]
        
        
        
    
    def gen_IG(self, c1_motion, c2_motion):
        
        c1_root_heading_quat,_ = self.get_headings(c1_motion)
        c2_root_heading_quat,_ = self.get_headings(c2_motion)
        keypoint_pos_c1_w = c1_motion[:,:66].reshape(-1,22,3)
        keypoint_pos_c2_w = c2_motion[:,:66].reshape(-1,22,3)
        
        IG = keypoint_pos_c1_w.unsqueeze(2) - keypoint_pos_c2_w.unsqueeze(1)
        IG = IG.reshape(IG.shape[0], -1, 3)
        
        IG_vel = IG[1:] - IG[:-1]
        
        # IG_pos_in_2 = keypoint_pos_c2_w.unsqueeze(2) - keypoint_pos_c1_w.unsqueeze(1)
        # IG_pos_in_2 = IG_pos_in_2.reshape(IG_pos_in_2.shape[0], -1, 3)
        
        # IG_in_1 = qrot(qinv(c1_root_heading_quat).unsqueeze(1).repeat(1, IG_pos_in_1.shape[1], 1), IG_pos_in_1)
        # IG_in_2 = qrot(qinv(c2_root_heading_quat).unsqueeze(1).repeat(1, IG_pos_in_2.shape[1], 1), IG_pos_in_2)
        
        # IG_in_1_vel = IG_in_1[1:] - IG_in_1[:-1]
        # IG_in_2_vel = IG_in_2[1:] - IG_in_2[:-1]
        
        # IG_in_1 = torch.cat([IG_in_1[:-1], IG_in_1_vel], dim=-1)
        # IG_in_2 = torch.cat([IG_in_2[:-1], IG_in_2_vel], dim=-1)
        
        IG = torch.cat([IG[:-1], IG_vel], dim=-1)
        
        return IG 
        
        
        
        

            
class TM2_phyTrain_dataset():
        def __init__(self, 
                     motion_name,
                     motion_length=100,
                     ref_horizon = 1,
                     dataset_path =TM2_ROOT_DIR+ '/motion_data/interGen/labeled/',
                     aig_path = TM2_ROOT_DIR+ '/results/saved/models/learned_aig.pt',
                     device='cpu'):
            self.data = {}
            self.device = device
            self.available_idx =None
            self.motion_length = motion_length
            self.ref_horizon = ref_horizon
            self.load_motion(motion_name, dataset_path, aig_path)
            
            
        def load_motion(self, motion_name, dataset_path, aig_path=None):
            # for key in ['c1_ori_motion','c2_ori_motion', 'clip_SE_idx']:
            #     self.data[key] = torch.load(dataset_path+motion_name+'.pt', map_location=self.device)[key]
            # if aig_path is None:
            #     aig_path = dataset_path
            # self.aig_idx = torch.load(aig_path,map_location=self.device)[motion_name]
            
            
            data = np.load(TM2_ROOT_DIR+'/results/saved/trajectories/dataset/{}.npy'.format(motion_name))
            self.data['c1_ori_motion'] = torch.tensor(data[0], device=self.device)
            self.data['c2_ori_motion'] = torch.tensor(data[1], device=self.device)
            self.aig_idx = torch.tensor([0, 20, 21, 20* 22, 21*22, 21*22-2, 21*22-1, 22*22-2,22*22-1]).to(device=self.device)
            self.aig_idx = torch.tensor([0, 21*22-1]).to(device=self.device)
            
            c1_key_pos = (self.data['c1_ori_motion'][:,:66]).reshape(-1,22,3)
            c2_key_pos = (self.data['c2_ori_motion'][:,:66]).reshape(-1,22,3)
            AIG_pos = (c1_key_pos.unsqueeze(2) - c2_key_pos.unsqueeze(1)).reshape(c1_key_pos.shape[0],-1, 3)[:, self.aig_idx]
            AIG_center =  ((c1_key_pos.unsqueeze(2) + c2_key_pos.unsqueeze(1))/2).reshape(c1_key_pos.shape[0],-1, 3)[:, self.aig_idx]

            
            self.data['AIG'] = torch.cat([AIG_pos, AIG_center], dim=-1).to(device=self.device)


        
        
        def sample(self,  sample_size, motion_length=None):
            # if motion_length is None:
            #     motion_length = self.motion_length
            # if self.available_idx is None or  motion_length!=self.motion_length:
            #     idx = self.data['clip_SE_idx']
            #     self.motion_length = motion_length
            #     self.available_idx = torch.cat([torch.arange(idx[i]+i, idx[i]+i+2000,250) for i in range(len(idx)-1)]).to(self.device)
            
            
            # # self.available_idx = torch.tensor(idx).to(self.device)
            # picked_idx = torch.randint(high=self.available_idx.shape[0], size=(sample_size,)).to(self.device)
            # clip_idx = self.available_idx[picked_idx] * 0 #+  self.data['clip_SE_idx'][1]+1
            return 0 #clip_idx
        
        

        

if __name__ == '__main__':    
    dataset = Motion_dataset(windows=10)
    # dataset.gen_dataset('sparring')   
    # dataset.gen_dataset('handshake') 
    # dataset.gen_dataset('rock-paper') 
    # dataset.gen_dataset('reach-hug-short') 
    # dataset.gen_dataset('hold-hand-and-circle')
    # dataset.gen_dataset('dancing')
    dataset.gen_dataset('fencing')
    # dataset.get_norm( ['sparring', 'handshake', 'rock-paper','reach-hug-short', 'hold-hand-and-circle'])    
    # dataset.load_and_form( ['sparring', 'handshake', 'rock-paper','reach-hug-short', 'hold-hand-and-circle']) 
    # for _ in range(10000):
    #     st1, nxtst1, st2, nxtst2, ig2 = dataset.sample(5, two_chars=True)
   


    