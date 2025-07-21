
from TM2_buddyImitation.runner.TM2_Runners import Interaction_Transfer
import TM2_buddyImitation.utils.helpers as helpers
from TM2_buddyImitation.utils.helpers import class_to_dict



def train(args):
    env_cfg = helpers.parse_env_cfg(args)
    train_cfg = helpers.parse_train_cfg(args)
    log_root_path, log_dir = helpers.get_log_dir(env_cfg, train_cfg, args)
    

    env = helpers.build_env(env_cfg, args)
    runner = Interaction_Transfer(env, class_to_dict(train_cfg), log_dir=log_dir)
    
    
    train_cfg.resume = False
    train_cfg.load_run = 'test'
    train_cfg.load_checkpoint = 'model_190000'
    
   
    # save resume path before creating a new log_dir
    helpers.load_runner_run(runner, log_root_path, train_cfg, env.device)
    helpers.load_primitives(runner, log_root_path, train_cfg, target_key='dec_0', source_key='dec_0', update_primitive=True)


    # set seed of the environment
    helpers.set_seed(train_cfg.seed)
    runner.learn(num_learning_iterations=train_cfg.max_iterations)




if __name__ == "__main__":
    args = helpers.get_args()
    train(args)