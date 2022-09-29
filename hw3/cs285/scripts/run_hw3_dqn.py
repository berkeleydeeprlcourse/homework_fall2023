import os
import time

from cs285.infrastructure.rl_trainer import RL_Trainer
from cs285.agents.dqn_agent import DQNAgent
from cs285.infrastructure.dqn_utils import get_env_kwargs


class Q_Trainer(object):

    def __init__(self, params):
        self.params = params

        train_args = {
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
            'num_critic_updates_per_agent_update': params['num_critic_updates_per_agent_update'],
            'train_batch_size': params['batch_size'],
            'double_q': params['double_q'],
        }

        env_args = get_env_kwargs(params['env_name'])

        self.agent_params = {**train_args, **env_args, **params}

        self.params['agent_class'] = DQNAgent
        self.params['agent_params'] = self.agent_params
        self.params['train_batch_size'] = params['batch_size']
        self.params['env_wrappers'] = self.agent_params['env_wrappers']

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):
        self.rl_trainer.run_training_loop(
            self.agent_params['num_timesteps'],
            collect_policy = self.rl_trainer.agent.actor,
            eval_policy = self.rl_trainer.agent.actor,
        )

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env_name',
        default='MsPacman-v0',
        choices=('PongNoFrameskip-v4', 'LunarLander-v3', 'MsPacman-v0')
    )

    parser.add_argument('--ep_len', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='todo')

    parser.add_argument('--eval_batch_size', type=int, default=1000)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--num_critic_updates_per_agent_update', type=int, default=1)
    parser.add_argument('--double_q', action='store_true')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--scalar_log_freq', type=int, default=int(1e4))
    parser.add_argument('--video_log_freq', type=int, default=-1)

    parser.add_argument('--save_params', action='store_true')

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)
    params['video_log_freq'] = -1 # This param is not used for DQN
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

    trainer = Q_Trainer(params)
    trainer.run_training_loop()


if __name__ == "__main__":
    main()
