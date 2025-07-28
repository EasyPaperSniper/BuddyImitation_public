import inspect
from TM2_buddyImitation import TM2_ENVS_DIR, TM2_ROOT_DIR
from TM2_buddyImitation.utils.configclass import BaseConfig




class MotionVAE_network(BaseConfig
):
    latent_dim = 32
    mvae_encoder_dims= [256, 256, 256]
    mvae_decoder_params= [8, 256, 256]
    activation="elu"
    model='VAE'
    


class MotionVAE_runner(BaseConfig):
    max_iterations = 1000
    learning_rate = 1.e-4
    save_interval = 200
    batch_size = 64
    kl_weight = 0.2
    

class MotionVAELearningCfg(BaseConfig):
    seed =1
    device = 'cuda:0'
    experiment_name = 'MotionVAE_learning'
    
    # logger
    log_dir = ''
    viz_tool = None
    log_interval = 50
    
    motion_list = ['sparring', 'handshake', 'rock-paper','reach-hug-short', 'hold-hand-and-circle']
    runner = MotionVAE_runner()
    network = MotionVAE_network()
    

class Embedding_network(BaseConfig):
    num_heads = 4
    trans_encoder_dims= [256, 256]
    ig_encoder_dims = [64,64]
    aig_encoder_dims= [256, 256]
    
class Embedding_Runner(BaseConfig):
    max_iterations = 1000
    learning_rate = 1.e-3
    save_interval = 200
    batch_size = 64
    saved_mvae_dir = TM2_ROOT_DIR + '/results/saved/models/mvae_May24.pt'
    



class EmbeddingLearningCfg(BaseConfig):
    
    experiment_name = 'embedding_learning'
    motion_list = ['sparring', 'handshake', 'rock-paper','reach-hug-short', 'circle']
    network = Embedding_network()
    runner = Embedding_Runner()
    
    
  