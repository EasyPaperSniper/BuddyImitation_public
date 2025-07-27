
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





if __name__ == '__main__':
    train_embedding()

