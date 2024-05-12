Assignments for [Berkeley CS 285: Deep Reinforcement Learning, Decision Making, and Control](http://rail.eecs.berkeley.edu/deeprlcourse/).

# Lecture 5.1 Policy Gradients and REINFORCE

Let the optimal weights be given by

$$
\theta^{*} = \underset{\theta}{argmax} E_{\tau \sim P_\theta(\tau)} [\underset{t}{\sum} r(s_t, a_t))]
$$

Let the value of weights be given by 

$$
J(\theta) = E_{\tau \sim P_\theta(\tau)} [\underset{t}{\sum} r(s_t, a_t)] = \int \nabla_{\theta} P_{\theta}(\tau) r(\tau) d\tau
$$

We can approximate $J(\theta)$ by sampling rollouts so 

$$
J(\theta) \approx \frac{1}{N} \sum_i \sum_t r(s_{i,t},a_{i,t})
$$

From basic calculus 

$$
P_{\theta}(\tau) \nabla_{\theta} log P_{\theta} (\tau) = P_{\theta} (\tau) \frac{\nabla_{\theta} P_{\theta}(\tau)}{P_{\theta}(\tau)} = \nabla_{\theta} P_{\theta}(\tau)
$$

Substituting the identity

$$
J(\theta) = \int \nabla_{\theta} P_{\theta}(\tau) r(\tau) d\tau
$$

$$
J(\theta) = \int [P_{\theta}(\tau) \nabla_{\theta} log P_{\theta} (\tau)] r(\tau) d\tau
$$

$$
J(\theta) = E_{\tau \sim P_\theta(\tau)} [\nabla_\theta log P_\theta(\tau) r(\tau)]
$$

Evaluate $\nabla_{\theta} log P_\theta(\tau)$ from the definition

$$
P_{\theta}(s_1, a_1, ..., s_T, a_T) = P(s_1) \overset{T}{\underset{t=1}{\prod}} \pi_{\theta}(a_t \vert s_t) p(s_{t+1} \vert  s_t, a_t)
$$

$$
log P_{\theta}(s_1, a_1, ..., s_T, a_T) = log P(s_1) + \overset{T}{\underset{t=1}{\sum}} [log \pi_\theta(a_t \vert s_t) + logP(s_{t+1} \vert s_t,a_t)]
$$

Cancelling the constant terms $\nabla_\theta logP(s_1) = 0$ and $\nabla_\theta log P(s_{t+1} \vert s_t,a_t) = 0$

$$
\nabla_{\theta} log P_{\theta}(\tau) =  \overset{T}{\underset{t=1}{\sum}}\nabla_\theta log \pi_\theta(a_t \vert s_t)
$$

Resubsituting to $J(\theta)$

$$
J(\theta) = E_{\tau \sim P_\theta(\tau)} [\nabla_\theta log P_\theta(\tau) r(\tau)]
$$

$$
J(\theta) = E_{\tau \sim P_\theta(\tau)} [ \overset{T}{\underset{t=1}{\sum}} \nabla_\theta log \pi_\theta(a_t \vert s_t) r(\tau)]
$$

$$
J(\theta) = E_{\tau \sim P_\theta(\tau)} [ (\overset{T}{\underset{t=1}{\sum}} \nabla_\theta log \pi_\theta(a_t \vert s_t)) (\overset{T}{\underset{t=1}{\sum}} r(\tau))]
$$

Then we can approximate $J(\theta)$ by sampling rollouts

$$
J(\theta) \approx \frac{1}{N} \overset{N}{\underset{i=1}{\sum}} [(\overset{T}{\underset{t=1}{\sum}} \nabla_\theta log \pi_\theta(a_{i,t} \vert s_{i,t})) (\overset{T}{\underset{t=1}{\sum}} r(s_{i,t},a_{i,t}))]
$$

Then we can update the network using the REINFORCE algorithm 

1)
	Sample $\{\tau^i\}$ from $\pi_{\theta}(a_t \vert s_t)$ by running policy
2)
	Evaluate $\nabla_\theta J(\theta) \approx \frac{1}{N} \overset{N}{\underset{i=1}{\sum}} [(\overset{T}{\underset{t=1}{\sum}} \nabla_\theta log \pi_\theta(a_{i,t} \vert s_{i,t})) (\overset{T}{\underset{t=1}{\sum}} r(s_{i,t},a_{i,t}))]$
3) Assign $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

# Lecture 5.2 

The derivation from 5.1 can be extended to the partial observable case to yield the following result

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \overset{N}{\underset{i=1}{\sum}} [(\overset{T}{\underset{t=1}{\sum}} \nabla_\theta log \pi_\theta(a_{i,t} \vert o_{i,t})) (\overset{T}{\underset{t=1}{\sum}} r(s_{i,t},a_{i,t}))]
$$

In practice policy gradient estimation suffers from high variance of collected rollouts. 

# Lecture 5.3



# 3.1 Behavioural Cloning

```

	expert_policy_file: 'cs285/policies/experts/Ant.pkl'
	expert_data: 'cs285/expert_data/expert_data_Ant-v4.pkl'
	env_name: Ant-v4
	exp_name: bc_ant
	do_dagger: false



```

| expert_policy_file               | expert_data                                | env_name | exp_name | do_dagger | ep_len |  |
| -------------------------------- | ------------------------------------------ | -------- | -------- | --------- | ------ | - |
| 'cs285/policies/experts/Ant.pkl' | 'cs285/expert_data/expert_data_Ant-v4.pkl' | Ant-v4   | bc_ant   | false     | 1000   |  |
|                                  |                                            |          |          |           |        |  |
