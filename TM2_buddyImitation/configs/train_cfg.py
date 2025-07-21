from TM2_buddyImitation.configs.env_cfg import BaseConfig




class BuddyImitationCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    empirical_normalization =  False
    
        
    num_steps_per_env = 25 # per iteration
    max_iterations = 10000 # number of policy updates

    # logging
    logger =None
    wandb_project = 'TM2'
    save_interval = 5000 # check for potential saves every this many iterations
    run_name = 'test'

    # load and resume
    resume = False
    load_run = -1 # -1 = last run
    load_checkpoint = -1 # -1 = last saved model
    resume_path = None # updated from load_run and chkpt



    class policy:
        class_name = 'TM2_ActorCritic'

        init_noise_std = 1.
        actor_latent_dim = 64
        actor_encoder_hidden_dims = [1024, 1024, 512]
        actor_decoder_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [1024, 1024, 512]
        activation="lrelu"
        share_primitive=True

        
    class algorithm:
        class_name = 'PPO'
 
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-4 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

