from collections import OrderedDict

from cs285.critics.dqn_critic import DQNCritic
from cs285.critics.cql_critic import CQLCritic
from cs285.critics.iql_critic import IQLCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.argmax_policy import ArgMaxPolicy
from cs285.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer
from cs285.exploration.rnd_model import RNDModel
from .dqn_agent import DQNAgent
from cs285.policies.MLP_policy import MLPPolicyAWAC
import numpy as np
import torch


class IQLAgent(DQNAgent):
    def __init__(self, env, agent_params, normalize_rnd=True, rnd_gamma=0.99):
        super(IQLAgent, self).__init__(env, agent_params)
        
        self.replay_buffer = MemoryOptimizedReplayBuffer(100000, 1, float_obs=True)
        self.num_exploration_steps = agent_params['num_exploration_steps']
        self.offline_exploitation = agent_params['offline_exploitation']

        self.exploitation_critic = IQLCritic(agent_params, self.optimizer_spec)
        self.exploration_critic = DQNCritic(agent_params, self.optimizer_spec)
        
        self.exploration_model = RNDModel(agent_params, self.optimizer_spec)
        self.explore_weight_schedule = agent_params['explore_weight_schedule']
        self.exploit_weight_schedule = agent_params['exploit_weight_schedule']
        
        self.actor = ArgMaxPolicy(self.exploitation_critic)
        self.eval_policy = self.awac_actor = MLPPolicyAWAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            self.agent_params['awac_lambda'],
        )

        self.exploit_rew_shift = agent_params['exploit_rew_shift']
        self.exploit_rew_scale = agent_params['exploit_rew_scale']
        self.eps = agent_params['eps']

        self.running_rnd_rew_std = 1
        self.normalize_rnd = normalize_rnd
        self.rnd_gamma = rnd_gamma

    def get_qvals(self, critic, obs, action=None, use_v=False):
        if use_v:
            q_value = critic.v_net(obs)
        else:
            qa_values = critic.q_net_target(obs)
            q_value = torch.gather(qa_values, 1, action.type(torch.int64).unsqueeze(1))
        return q_value

    def estimate_advantage(self, ob_no, ac_na, re_n, next_ob_no, terminal_n, n_actions=10):
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        re_n = ptu.from_numpy(re_n)
        next_ob_no = ptu.from_numpy(next_ob_no)
        terminal_n = ptu.from_numpy(terminal_n)

        v_pi = self.get_qvals(self.exploitation_critic, ob_no, use_v=True)
        return self.get_qvals(self.exploitation_critic, ob_no, ac_na) - v_pi

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}

        if self.t > self.num_exploration_steps:
            # TODO: After exploration is over, set the actor to optimize the extrinsic critic
            #HINT: Look at method ArgMaxPolicy.set_critic

        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):

            # TODO: Get Reward Weights
            # Get the current explore reward weight and exploit reward weight
            explore_weight = None
            exploit_weight = None 

            # TODO: Run Exploration Model #
            # Evaluate the exploration model on s to get the exploration bonus
            # HINT: Normalize the exploration bonus, as RND values vary highly in magnitude
            expl_bonus = None

            # TODO: Reward Calculations #
            # Calculate mixed rewards, which will be passed into the exploration critic
            # HINT: See doc for definition of mixed_reward
            mixed_reward = None

            # TODO: Calculate the environment reward
            # HINT: For part 1, env_reward is just 're_n'
            #       After this, env_reward is 're_n' shifted by self.exploit_rew_shift,
            #       and scaled by self.exploit_rew_scale
            env_reward = None

            # TODO: Update Critics And Exploration Model #
            # 1): Update the exploration model (based off s')
            # 2): Update the exploration critic (based off mixed_reward)
            # 3): a) Update the exploitation critic's Value function
            # 3): b) Update the exploitation critic's Q function (based off env_reward)
            expl_model_loss = None
            exploration_critic_loss = None 
            exploitation_critic_loss = None
            exploitation_critic_loss.update(TODO)



            # TODO: update actor as in AWAC
            # 1): Estimate the advantage
            # 2): Calculate the awac actor loss
            advantage = None
            actor_loss = None

            # TODO: Update Target Networks #
            if self.num_param_updates % self.target_update_freq == 0:
                #  Update the exploitation and exploration target networks
                pass

            # Logging #
            log['Exploration Critic Loss'] = exploration_critic_loss['Training Loss']
            log['Exploitation Critic V Loss'] = exploitation_critic_loss['Training Q Loss']
            log['Exploitation Critic Q Loss'] = exploitation_critic_loss['Training V Loss']
            log['Exploration Model Loss'] = expl_model_loss

            # <DONE>: Uncomment these lines after completing awac
            # log['Actor Loss'] = actor_loss

            self.num_param_updates += 1

        self.t += 1
        return log


    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """
        if (not self.offline_exploitation) or (self.t <= self.num_exploration_steps):
            self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

        perform_random_action = np.random.random() < self.eps or self.t < self.learning_starts

        if perform_random_action:
            action = self.env.action_space.sample()
        else:
            processed = self.replay_buffer.encode_recent_observation()
            action = self.actor.get_action(processed)

        next_obs, reward, done, info = self.env.step(action)
        self.last_obs = next_obs.copy()

        if (not self.offline_exploitation) or (self.t <= self.num_exploration_steps):
            self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        if done:
            self.last_obs = self.env.reset()
