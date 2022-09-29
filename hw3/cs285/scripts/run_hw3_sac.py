import os
import time

from cs285.agents.sac_agent import SACAgent
from cs285.infrastructure.rl_trainer import RL_Trainer


class SAC_Trainer(object):

    def __init__(self, params):

        #####################
        ## SET AGENT PARAMS
        #####################

        computation_graph_args = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
            'init_temperature': params['init_temperature'],
            'actor_update_frequency': params['actor_update_frequency'],
            'critic_target_update_frequency': params['critic_target_update_frequency']
            }

        estimate_advantage_args = {
            'gamma': params['discount'],
        }

        train_args = {
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
            'num_critic_updates_per_agent_update': params['num_critic_updates_per_agent_update'],
            'num_actor_updates_per_agent_update': params['num_actor_updates_per_agent_update'],
        }

        agent_params = {**computation_graph_args, **estimate_advantage_args, **train_args}

        self.params = params
        self.params['agent_class'] = SACAgent
        self.params['agent_params'] = agent_params
        self.params['batch_size_initial'] = self.params['batch_size']

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):
        self.rl_trainer.run_sac_training_loop(
            self.params['n_iter'],
            collect_policy = self.rl_trainer.agent.actor,
            eval_policy = self.rl_trainer.agent.actor,
            )


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='CartPole-v0')
    parser.add_argument('--ep_len', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='todo')
    parser.add_argument('--n_iter', '-n', type=int, default=200)

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--num_critic_updates_per_agent_update', type=int, default=1)
    parser.add_argument('--num_actor_updates_per_agent_update', type=int, default=1)
    parser.add_argument('--actor_update_frequency', type=int, default=1)
    parser.add_argument('--critic_target_update_frequency', type=int, default=1)
    parser.add_argument('--batch_size', '-b', type=int, default=1000) #steps collected per train iteration
    parser.add_argument('--eval_batch_size', '-eb', type=int, default=400) #steps collected per eval iteration
    parser.add_argument('--train_batch_size', '-tb', type=int, default=256) ##steps used per gradient step

    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--init_temperature', '-temp', type=float, default=1.0)
    parser.add_argument('--learning_rate', '-lr', type=float, default=3e-4)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--video_log_freq', type=int, default=-1)
    parser.add_argument('--scalar_log_freq', type=int, default=10)

    parser.add_argument('--save_params', action='store_true')

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    ###################
    ### RUN TRAINING
    ###################

    trainer = SAC_Trainer(params)
    trainer.run_training_loop()


if __name__ == "__main__":
    main()
