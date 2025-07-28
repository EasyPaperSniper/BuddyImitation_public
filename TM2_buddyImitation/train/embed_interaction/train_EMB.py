
import torch.nn as nn

from TM2_buddyImitation import TM2_ROOT_DIR
from TM2_buddyImitation.runner.TM2_Runners import Embeding_Trainer
from TM2_buddyImitation.runner.learning_module import Graph_Embedding, MotionVAE
from TM2_buddyImitation.configs.EIG_config import EmbeddingLearningCfg


from TM2_buddyImitation.utils.quaterion import *
from TM2_buddyImitation.utils.ratation_conversion import *
from TM2_buddyImitation.utils.process_intergen.interGen_param import *


def train_embedding():
   
    train_cfg = EmbeddingLearningCfg().to_dict()
    train_cfg["log_dir"] = TM2_ROOT_DIR + '/results/embedding/'
    
    
    motion_data = Motion_dataset(device=train_cfg['device'])
    motion_data.load_and_form(train_cfg["motion_list"])
    
    trainer = Embeding_Trainer(train_cfg, dataset = motion_data, device=train_cfg['device'])
    print('********************Training Embeded Graph ********************')
    trainer.train(num_learning_iterations= train_cfg['runner']['max_iterations'])





if __name__ == '__main__':
    train_embedding()

