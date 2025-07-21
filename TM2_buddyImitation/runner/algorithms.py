import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.algorithms import PPO


class TM2_PPO(PPO):
    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
                
                
            # High level regularzation
            high_level_output  = self.actor_critic.encoder_distribution
            high_level_mean, high_level_sigma = torch.mean(high_level_output, dim=0), torch.std(high_level_output, dim=0)
            high_level_kl = torch.sum(
                        torch.log(1 / high_level_sigma + 1.0e-5)
                        + (torch.square(high_level_sigma) + torch.square(high_level_mean ))/2,dim=-1)
            
            

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + 0.1*high_level_kl

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss


class MotionVAELearning:
    def __init__(self,
                    MotionVAE_net,
                    dataset,
                    learning_rate=1e-3,
                    kl_weight=0.5,
                    device='cpu',
                    **kwargs):
        self.device = device
        self.learning_rate = learning_rate
        self.MotionVAE_net = MotionVAE_net.to(self.device)
        self.dataset = dataset
        self.VAE_optimizer = optim.Adam(self.MotionVAE_net.parameters(), lr=learning_rate)
        self.MSE_loss = nn.MSELoss()
        self.step = 0
        self.beta = kl_weight

        
    def update_network(self, num_epochs=1, epoch_idx=0, batch_size=64):
        total_mse_loss, total_loss = 0, 0
        for epoch in range(num_epochs):
            s_t, s_tp1 = self.dataset.sample(batch_size)
            s_t, s_tp1 = s_t.to(self.device), s_tp1.to(self.device)
            
            if epoch_idx < 200:
                # supervised learning
                s_t = s_t.reshape(-1, s_t.shape[-1])
                s_tp1 = s_tp1.reshape(-1, s_tp1.shape[-1])
                s_tp1_pred, latent_ori, latent_dec, latent_enc_for_embd = self.MotionVAE_net.forward(s_t, s_tp1)
                prediction = s_tp1_pred
                latent_var = latent_ori
            else :
                # scheduled sampling
                windows = s_t.shape[1]
                s_t_cur = s_t[:,0,:]
                s_tp1_cur = s_tp1[:,0,:]

                s_tp1_pred, latent_ori, latent_dec, latent_enc_for_embd = self.MotionVAE_net.forward(s_t_cur, s_tp1_cur)
                prediction = [s_tp1_pred]
                lat_pred = [latent_ori]
                
                for i in range(1, windows):
                    if np.random.rand() > (epoch_idx-200)/300:
                        s_t_cur = s_t[:,i,:]
                    else:
                        s_t_cur = s_tp1_pred
                    s_tp1_cur = s_tp1[:,i,:]
                    s_tp1_pred, latent_ori, latent_dec, latent_enc_for_embd = self.MotionVAE_net.forward(s_t_cur, s_tp1_cur)
                    prediction.append(s_tp1_pred)
                    lat_pred.append(latent_ori)
                prediction = torch.cat([x[:,None] for x in prediction], dim=1)
                latent_var = torch.cat([x for x in lat_pred], dim=0)
                    

            mse_loss = self.MSE_loss(prediction, s_tp1) 
            

            if self.MotionVAE_net.model == 'VAE':
                kl_loss =  self.cal_kl_loss(latent_var)
                loss = mse_loss + kl_loss
            if self.MotionVAE_net.model =='VQVAE':
                z_and_sg_embd_loss = self.MSE_loss(latent_ori,latent_dec.detach())
                sg_z_and_embd_loss = self.MSE_loss(self.MotionVAE_net.codebook,
                                                   latent_enc_for_embd.detach())
                loss = mse_loss + sg_z_and_embd_loss+ self.beta*z_and_sg_embd_loss 


            self.VAE_optimizer.zero_grad()
            loss.backward()
            self.VAE_optimizer.step()
            
            total_mse_loss += mse_loss.item()
            total_loss += loss.item()
            self.step += 1
            
        total_mse_loss /= num_epochs
        total_loss /= num_epochs
        
        return total_mse_loss, total_loss
    

    def cal_kl_loss(self, latent_var):
        mu = torch.mean(latent_var, dim=0)
        logvar = torch.log(torch.var(latent_var, dim=0))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())* self.beta
        return kl_loss



class GraphEmbeddingLearning:
    def __init__(self,
                 embedding_net,
                 MotionVAE_net,
                 dataset,
                 learning_rate=1e-3,
                 device='cpu',
                 **kwargs) -> None:
        
        self.device = device
        self.learning_rate = learning_rate
        self.embedding_net = embedding_net.to(self.device)
        self.MotionVAE_net = MotionVAE_net.to(self.device)
        self.dataset = dataset
        self.emb_optimizer = optim.Adam(self.embedding_net.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.step = 0
        if self.embedding_net.predefine_graph is not None:
            self.qk_hard_tgt = torch.zeros((1, 1,1, self.embedding_net.ig_dim[0] )).to(self.device)
            for i in range(self.embedding_net.predefine_graph.shape[0]):
                # if i< self.embedding_net.num_heads:
                self.qk_hard_tgt[:,0,0,self.embedding_net.predefine_graph[i]] = 1

   
    def update_network(self, num_epochs=1, epoch_idx=0, batch_size=32):
        mean_loss,reconst_loss_sum = 0, 0
        
        for _ in range(num_epochs):
            if epoch_idx< 200:
                c1_st, c1_stp1, c2_st, c2_stp1, ig = self.dataset.sample(batch_size, two_chars=True)

                c1_st = c1_st.reshape(-1, c1_st.shape[-1])
                c1_stp1 = c1_stp1.reshape(-1, c1_stp1.shape[-1])
                c2_st = c2_st.reshape(-1, c2_st.shape[-1])
                c2_stp1 = c2_stp1.reshape(-1, c2_stp1.shape[-1])
                ig = ig.reshape(-1, ig.shape[-2], ig.shape[-1])
            
                qk,  qk_hard, att_out, c1_lat, c2_lat = self.embedding_net.predict(c1_st, c2_st, ig)
                
                pred_c1_stp1 = self.MotionVAE_net.decode(c1_st, c1_lat)
                pred_c2_stp1 = self.MotionVAE_net.decode(c2_st, c2_lat)

                
                reconst_loss = (self.loss(pred_c1_stp1, c1_stp1) + self.loss(pred_c2_stp1, c2_stp1)) 
                loss = reconst_loss + 10*torch.var(qk_hard)
                
                if self.embedding_net.predefine_graph is not None:
                    loss += 10* self.loss(torch.sum(qk_hard,dim=1,keepdim=True), self.qk_hard_tgt.repeat(qk_hard.shape[0],1,1,1)) 
                
            else:
                c1_st, c1_stp1, c2_st, c2_stp1, ig = self.dataset.sample(batch_size, two_chars=True)
                
                windows = c1_st.shape[1]
                c1_st_cur, c2_st_cur = c1_st[:,0,:], c2_st[:,0,:]
                ig_cur = ig[:,0,:]
                
                qk,  qk_hard, att_out, c1_lat, c2_lat = self.embedding_net.predict(c1_st_cur, c2_st_cur, ig_cur)
                soft_out = [qk]
                hard_out = [qk_hard]
                c1_pred_stp1 = [self.MotionVAE_net.decode(c1_st_cur, c1_lat)]
                c2_pred_stp1 = [self.MotionVAE_net.decode(c2_st_cur, c2_lat)]

                
                for i in range(1, windows):
                    if np.random.rand() < (epoch_idx-200)/300:
                        c1_st_cur, c2_st_cur = c1_st[:,i,:], c2_st[:,i,:]
                    else:
                        c1_st_cur, c2_st_cur = c1_pred_stp1[-1], c2_pred_stp1[-1]

                    ig_cur = ig[:,i,:]
                    qk,  qk_hard, att_out, c1_lat, c2_lat = self.embedding_net.predict(c1_st_cur, c2_st_cur, ig_cur)
                    soft_out.append(qk)
                    hard_out.append(qk_hard)
                    c1_pred_stp1.append(self.MotionVAE_net.decode(c1_st_cur, c1_lat))
                    c2_pred_stp1.append(self.MotionVAE_net.decode(c2_st_cur, c2_lat))

                
                c1_pred_stp1 = torch.cat([x[:,None] for x in c1_pred_stp1], dim=1)
                c2_pred_stp1 = torch.cat([x[:,None] for x in c2_pred_stp1], dim=1)
                soft_out = torch.cat([x[:,None] for x in soft_out], dim=1)
                hard_out = torch.cat([x for x in hard_out], dim=0)
                
                                                  
                reconst_loss = (self.loss(c1_stp1, c1_pred_stp1) + self.loss(c2_stp1, c2_pred_stp1)) 
                loss = reconst_loss + 10*torch.var(hard_out) 
                
                if self.embedding_net.predefine_graph is not None:
                    loss += 10* self.loss(torch.sum(hard_out,dim=1,keepdim=True), self.qk_hard_tgt.repeat(hard_out.shape[0],1,1,1)) 
                   
            self.emb_optimizer.zero_grad()
            loss.backward()
            self.emb_optimizer.step()
            
            reconst_loss_sum += reconst_loss.item()
            mean_loss += loss.item()
            self.step += 1
        mean_loss /= num_epochs
        reconst_loss = reconst_loss_sum/num_epochs
        return mean_loss, reconst_loss

   
    def get_nxt_ig(self):
        pass
    


class XmorphLearning:
    def __init__(self, 
                 Xmorph_net,
                 device='cpu',
                 num_learning_epochs=1,
                num_mini_batches=1,
                learning_rate=1e-4,
                **kwargs):
        
        self.device = device
        self.Xmorph_net = Xmorph_net.to(device=device)
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.Xmorph_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()
        
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        
    def init_storage(self, num_envs, num_storage, input_shape, output_shape,):
        self.storage = SLstorage( num_storage,num_envs, input_shape, output_shape, self.device)

    def store(self, input, tgt):
        self.storage.add_data(input.detach(), tgt.detach())

    def update(self,):
        mean_reconst_loss =  0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for input_batch, tgt_batch in generator:
            
            scr2tgt = self.Xmorph_net.forward(input_batch)
            forward_loss = self.loss_func(scr2tgt, tgt_batch)
            tgt2scr = self.Xmorph_net.inverse(tgt_batch)
            inverse_loss = self.loss_func(tgt2scr, input_batch)
            loss = forward_loss+inverse_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
          
            mean_reconst_loss += loss.item()


        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_reconst_loss /= num_updates
        self.storage.clear()
        return mean_reconst_loss
        




class SLstorage():
    def __init__(self,
                 num_storage, num_envs, input_shape, output_shape, device='cpu') -> None:
        self.num_storage = num_storage
        self.num_envs = num_envs
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.device = device


        self.inputs = torch.zeros(( num_storage, num_envs, *input_shape), dtype=torch.float32, device=device)
        self.tgts = torch.zeros((  num_storage, num_envs,*output_shape), dtype=torch.float32, device=device)

        self.clear()


    def add_data(self, input, tgt):
        self.inputs[self.step].copy_(input)
        self.tgts[self.step].copy_(tgt)
        self.step = (self.step + 1) % self.num_storage

    def clear(self):
        self.step = 0
    

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_storage
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)


        inputs = self.inputs.flatten(0, 1)
        tgts = self.tgts.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                input_batch = inputs[batch_idx]
                tgt_batch = tgts[batch_idx]

                yield input_batch, tgt_batch