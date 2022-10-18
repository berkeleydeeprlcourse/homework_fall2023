import os
import time

from cs285.infrastructure.rl_trainer import RL_Trainer
from cs285.agents.mbpo_agent import MBPOAgent


class MBPO_Trainer(object):

    def __init__(self, params):

        

        #####################
        ## SET AGENT PARAMS
        #####################

        mb_computation_graph_args = {
            'ensemble_size': params['ensemble_size'],
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
        }
        
        sac_computation_graph_args = {
            'n_layers': params['sac_n_layers'],
            'size': params['sac_size'],
            'learning_rate': params['sac_learning_rate'],
            'init_temperature': params['sac_init_temperature'],
            'actor_update_frequency': params['sac_actor_update_frequency'],
            'critic_target_update_frequency': params['sac_critic_target_update_frequency']
        }
        
        mb_train_args = {
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
        }

        sac_train_args = {
            'num_agent_train_steps_per_iter': params['sac_num_agent_train_steps_per_iter'],
            'num_critic_updates_per_agent_update': params['sac_num_critic_updates_per_agent_update'],
            'num_actor_updates_per_agent_update': params['sac_num_actor_updates_per_agent_update'],
            'n_iter': params['sac_n_iter'],
            'train_batch_size': params['sac_train_batch_size']
        }

        estimate_advantage_args = {
            'gamma': params['sac_discount'],
        }

        controller_args = {
            'mpc_horizon': params['mpc_horizon'],
            'mpc_num_action_sequences': params['mpc_num_action_sequences'],
            'mpc_action_sampling_strategy': params['mpc_action_sampling_strategy'],
            'cem_iterations': params['cem_iterations'],
            'cem_num_elites': params['cem_num_elites'],
            'cem_alpha': params['cem_alpha'],
        }

        mb_agent_params = {**mb_computation_graph_args, **mb_train_args, **controller_args}
        sac_agent_params = {**sac_computation_graph_args, **estimate_advantage_args, **sac_train_args}
        agent_params = {**mb_agent_params}
        agent_params['sac_params'] = sac_agent_params

        self.params = params
        self.params['agent_class'] = MBPOAgent
        self.params['agent_params'] = agent_params

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):

        self.rl_trainer.run_training_loop(
            self.params['n_iter'],
            collect_policy = self.rl_trainer.agent.actor,
            eval_policy = self.rl_trainer.agent.actor,
            )


def main():

    import argparse
    parser = argparse.ArgumentParser()
    # Common parameters
    parser.add_argument('--env_name', type=str) #reacher-cs285-v0, ant-cs285-v0, cheetah-cs285-v0, obstacles-cs285-v0
    parser.add_argument('--ep_len', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='todo')
    parser.add_argument('--n_iter', '-n', type=int, default=20)
    parser.add_argument('--eval_batch_size', '-eb', type=int, default=400) #steps collected per eval iteration
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--video_log_freq', type=int, default=-1) #-1 to disable
    parser.add_argument('--scalar_log_freq', type=int, default=1) #-1 to disable
    parser.add_argument('--save_params', action='store_true')

    # Model learning parameters
    parser.add_argument('--train_batch_size', '-tb', type=int, default=512) #steps used per gradient step (used for training)
    parser.add_argument('--ensemble_size', type=int, default=3)
    parser.add_argument('--mpc_horizon', type=int, default=10)
    parser.add_argument('--mpc_num_action_sequences', type=int, default=1000)
    parser.add_argument('--mpc_action_sampling_strategy', type=str, default='random')
    parser.add_argument('--cem_iterations', type=int, default=4)
    parser.add_argument('--cem_num_elites', type=int, default=5)
    parser.add_argument('--cem_alpha', type=float, default=1)
    parser.add_argument('--add_sl_noise', action='store_true')
    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000)
    parser.add_argument('--batch_size_initial', type=int, default=20000) #(random) steps collected on 1st iteration (put into replay buffer)
    parser.add_argument('--batch_size', type=int, default=8000) #steps collected per train iteration (put into replay buffer)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--size', type=int, default=250)

    # SAC parameters
    parser.add_argument('--sac_num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--sac_num_critic_updates_per_agent_update', type=int, default=1)
    parser.add_argument('--sac_num_actor_updates_per_agent_update', type=int, default=1)
    parser.add_argument('--sac_actor_update_frequency', type=int, default=1)
    parser.add_argument('--sac_critic_target_update_frequency', type=int, default=1)
    parser.add_argument('--sac_train_batch_size', type=int, default=256) ##steps used per gradient step
    parser.add_argument('--sac_batch_size', type=int, default=1000) #steps collected per train iteration
    parser.add_argument('--sac_discount', type=float, default=0.99)
    parser.add_argument('--sac_init_temperature', type=float, default=1.0)
    parser.add_argument('--sac_learning_rate', type=float, default=3e-4)
    parser.add_argument('--sac_n_layers', type=int, default=2)
    parser.add_argument('--sac_size', type=int, default=64)
    parser.add_argument('--sac_n_iter', type=int, default=200)

    # MBPO parameters
    parser.add_argument('--mbpo_rollout_length', type=int, default=1)

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    # HARDCODE EPISODE LENGTHS FOR THE ENVS USED IN THIS MB ASSIGNMENT
    if params['env_name']=='reacher-cs285-v0':
        params['ep_len']=200
    if params['env_name']=='cheetah-cs285-v0':
        params['ep_len']=500
    if params['env_name']=='obstacles-cs285-v0':
        params['ep_len']=100

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    logdir_prefix = 'hw4_'  # keep for autograder

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    ###################
    ### RUN TRAINING
    ###################

    trainer = MBPO_Trainer(params)
    trainer.run_training_loop()


if __name__ == "__main__":
    main()
