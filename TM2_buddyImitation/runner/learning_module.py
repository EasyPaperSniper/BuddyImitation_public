import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.distributions import Normal
import einops
from einops.layers.torch import Rearrange
from torch.nn.parameter import Parameter

from TM2_buddyImitation.runner.utils import *




class TM2_ActorCritic(nn.Module):
    is_recurrent=False
    def __init__(
        self,
        num_actor_obs_list,
        num_critic_obs,
        num_actions_list,
        actor_latent_dim = 64,
        actor_encoder_hidden_dims = [512, 512],
        actor_decoder_hidden_dims = [512, 512, 512],
        critic_hidden_dims= [1024, 1024, 1024],
        activation="relu",
        init_noise_std=1.0,
        share_primitive=True,
        **kwargs,
    ):

        super().__init__()
        self.share_primitive = share_primitive
        
        activation = get_activation(activation)

        self.num_actor_obs_list = num_actor_obs_list
        self.num_actions = num_actions_list
        self.latent_idx = actor_latent_dim * torch.arange(0, len(num_actions_list)+1) # idx of latent for each dec
        self.obs_idx = np.insert(np.cumsum(self.num_actor_obs_list) ,0,0)  


        self.actor = nn.ModuleDict()
        self.actor['enc'] = nn.Sequential(*build_mlp(
                                                    sum(num_actor_obs_list), 
                                                    len(num_actions_list)*actor_latent_dim, 
                                                    actor_encoder_hidden_dims, 
                                                    activation))

        for i in range(len(num_actions_list)):
            self.actor['dec_'+str(i)] = nn.Sequential(*build_mlp(
                                                                num_actor_obs_list[i]+actor_latent_dim, 
                                                                num_actions_list[i], 
                                                                actor_decoder_hidden_dims, 
                                                                activation))

        
        self.critic = nn.Sequential(*build_mlp(num_critic_obs, 1, critic_hidden_dims, activation))
        
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(sum(num_actions_list)))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False



    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]


    def reset(self, dones=None): 
        pass


    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def forward(self, observations):
        latent = self.actor['enc'](observations)
        output = []
        for i in range(len(self.num_actions)):
            sub_obs = observations[:, self.obs_idx[i]:self.obs_idx[i+1]]
            sub_lat = latent[:, self.latent_idx[i]: self.latent_idx[i+1]]
            sub_policy_obs = torch.cat([sub_obs, sub_lat], dim=-1)
            if self.share_primitive:
                output.append(self.actor['dec_0'](sub_policy_obs))
            else:
                output.append(self.actor['dec_'+str(i)](sub_policy_obs))
        return torch.cat(output,dim=-1)


    def update_distribution(self, observations):
        mean = self.forward(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)
        self.encoder_distribution =  self.actor['enc'](observations)


    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.forward(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value





class Graph_Embedding(nn.Module):
    def __init__(
        self,
        obs_dim,
        ig_dim,
        latent_dim = 32,
        num_heads = 4,
        trans_encoder_dims= [512, 256],
        ig_encoder_dims= [256, 256],
        aig_encoder_dims= [512, 512],
        activation="relu",
        predefine_graph = None,
        **kwargs,
    ):
        super().__init__()
        self.scale = latent_dim ** -0.5
        activation = get_activation(activation)
        hidden_dim = latent_dim * num_heads
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.ig_dim = ig_dim


        self.to_k = nn.Conv1d(latent_dim, hidden_dim, 1, bias=False)
        self.to_q = nn.Conv1d(latent_dim, hidden_dim, 1, bias=False)
        self.out = nn.Conv1d(hidden_dim, latent_dim, 1)

        self.q_Layernorm = LayerNorm(latent_dim)
        self.k_Layernorm = LayerNorm(latent_dim)

        self.trans_encoder = nn.Sequential(*build_mlp(obs_dim*2, latent_dim, trans_encoder_dims, activation))
        self.ig_encoder = nn.Sequential(*build_mlp(ig_dim[-1], latent_dim, ig_encoder_dims, activation))
        self.aig_encoder = nn.Sequential(*build_mlp(num_heads*ig_dim[-1], latent_dim*2, aig_encoder_dims, activation))      
        

        self.predefine_graph = predefine_graph
        
    
    def predict(self, c1_obs,c2_obs, ig):
        obs = torch.cat([c1_obs, c2_obs], dim=-1 )
        
        trans_enc = self.trans_encoder(obs).unsqueeze(-1)
        trans_enc = self.q_Layernorm(trans_enc)
        q = einops.rearrange(self.to_q(trans_enc), 'b (h c) d -> b h c d', h=self.num_heads)
        v = (einops.rearrange(ig, 'b c d -> b d c').unsqueeze(1)).repeat(1, self.num_heads, 1, 1)


        ig_flat = einops.rearrange(ig, 'b c d -> (b c) d')
        ig_emb = einops.rearrange(self.ig_encoder(ig_flat), '(b c) d -> b c d', b = v.shape[0])
        ig_emb = einops.rearrange(ig_emb, 'b c d -> b d c')
        ig_emb = self.k_Layernorm(ig_emb) 
        k = einops.rearrange(self.to_k(ig_emb), 'b (h c) d -> b h c d', h=self.num_heads)


        qk = torch.einsum('b h d e, b h d n -> b h e n', q, k)* self.scale
        qk_soft = F.gumbel_softmax(qk, hard=False, dim = -1)
        qk_hard = F.gumbel_softmax(qk, hard=True, dim = -1) 
        qk_hard[:,0] = qk_hard[:,0]*0
        qk_hard[:,0, 0, 0] = 1
        att_out = torch.einsum('b h d n, b h e n -> b h d e', v, qk_hard)
        att_out = einops.rearrange(att_out, 'b h c d -> b (h c d)')
            
        # if self.predefine_graph is not None:
        #     att_out = ig[:, self.predefine_graph].reshape(ig.shape[0], -1)
        #     non_use_head = self.num_heads-np.shape(self.predefine_graph)[0]
        #     att_out = torch.cat([att_out, torch.zeros(ig.shape[0],self.ig_dim[-1] * non_use_head).to(ig.device)], dim = 1)
            
        # lat_var = self.aig_encoder(torch.cat([c1_obs, c2_obs, att_out],dim=-1))
        # att_out = ig.reshape(ig.shape[0], -1)
        lat_var = self.aig_encoder(att_out)
        c1_lat = lat_var[:, :self.latent_dim]
        c2_lat = lat_var[:, self.latent_dim:]

        
        return qk_soft,  qk_hard, att_out, c1_lat, c2_lat
    

    
    def load(self, load_path, device= None):
        if device is None:
            self.load_state_dict(torch.load(load_path)['emb_net_model'])
        else:    
            self.load_state_dict(torch.load(load_path, map_location=device)['emb_net_model'])
    



class MotionVAE(nn.Module):
    def __init__(
        self,
        obs_dim,
        nxt_obs_dim,
        latent_dim = 32,
        mvae_encoder_dims= [512, 256],
        mvae_decoder_params= [4, 256, 256],
        model='VAE', # VQVAE or VAE
        activation="elu",
        **kwargs,
    ):
        super().__init__()
        activation = get_activation(activation)
        num_experts = mvae_decoder_params[0]
        hg = mvae_decoder_params[1]
        h = mvae_decoder_params[2]
        self.encoder = nn.Sequential(*build_mlp(obs_dim+nxt_obs_dim, latent_dim, mvae_encoder_dims, activation))
        self.decoder = MANN_network(obs_dim, nxt_obs_dim, latent_dim, num_experts, hg, h, activation)
        self.model = model
        if model == 'VQVAE':
            self.codebook = nn.Parameter(torch.empty(mvae_decoder_params[3], latent_dim))
            torch.nn.init.uniform_(self.codebook, -1, 1)
       
        
    def encode(self, obs, nxt_obs):
        return self.encoder(torch.cat([obs, nxt_obs], dim=-1))

    
    def decode(self, obs, latent):
        return self.decoder(obs, latent)


    def forward(self, obs, nxt_obs):
        if self.model == 'VAE':
            latent = self.encode(obs, nxt_obs)
            output = self.decode(obs, latent)
            return output, latent, None, None

        if self.model == 'VQVAE':
            latent_ori = self.encode(obs, nxt_obs)
            latent_dec = self.find_nearest(latent_ori, self.codebook)
            latent_dec.register_hook(self.hook)
            output = self.decode(obs, latent_dec)
            latent_enc_for_embd = self.find_nearest(self.codebook, latent_ori)

            return output, latent_ori, latent_dec, latent_enc_for_embd


    def find_nearest(self, x, y):
        Q = x.unsqueeze(1).repeat(1, y.shape[0], 1)
        T = y.unsqueeze(0).repeat(x.shape[0], 1, 1)
        index=(Q-T).pow(2).sum(2).sqrt().min(1)[1]
        return y[index]
    
    def load(self, load_path, device= None):
        if device is None:
            self.load_state_dict(torch.load(load_path)['mvae_net_model'])
        else:
            self.load_state_dict(torch.load(load_path, map_location=device)['mvae_net_model'])
        self.eval()

    def hook(self, grad):
        self.grad_for_encoder = grad
        return grad




class Xmorph_network(nn.Module):
    def __init__(
        self,
        src_dim,
        tgt_dim,
        mapper_dims= [512, 128, 512],
        activation = 'lrelu',
        **kwargs
    ):
        super().__init__()
        activation = get_activation(activation)
        self.src2tgt_mapper = nn.Sequential(*build_mlp(src_dim, tgt_dim, mapper_dims, activation))
        self.tgt2src_mapper = nn.Sequential(*build_mlp(tgt_dim, src_dim, mapper_dims, activation))

    def forward(self, input):
        return self.src2tgt_mapper(input)
    
    def inverse(self, input):
        return self.tgt2src_mapper(input)
    
    def cycle_forward(self,input):
        mid_output = self.forward(input)
        output = self.inverse(mid_output)
        return mid_output, output




