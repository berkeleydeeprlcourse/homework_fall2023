## Setup

You can run this code on your own machine or on Google Colab. 

1. **Local option:** If you choose to run locally, you will need to install MuJoCo and some Python packages; see [installation.md](../hw1/installation.md) from homework 1 for instructions. There are two new package requirements (`opencv-python` and `gym[atari]`) beyond what was used in the previous assignments; make sure to install these with `pip install -r requirements.txt` if you are running the assignment locally.

2. **Colab:** The first few sections of the notebook will install all required dependencies. You can try out the Colab option by clicking the badges below:


RND + CQL:

[![Open In Colab RND + CQL](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berkeleydeeprlcourse/homework_fall2022/blob/master/hw5/run_hw5_expl.ipynb)

AWAC:

[![Open In Colab AWAC](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berkeleydeeprlcourse/homework_fall2022/blob/master/hw5/run_hw5_awac.ipynb)

IQL:

[![Open In Colab IQL](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berkeleydeeprlcourse/homework_fall2022/blob/master/hw5/run_hw5_iql.ipynb)

## Complete the code

The following files have blanks to be filled with your solutions from homework 1 and 3. The relevant sections are marked with `TODO'. You can get solutions from Ed. 

- [infrastructure/utils.py](cs285/infrastructure/utils.py)
- [infrastructure/rl_trainer.py](cs285/infrastructure/rl_trainer.py)
- [policies/MLP_policy.py](cs285/policies/MLP_policy.py)
- [policies/argmax_policy.py](cs285/policies/argmax_policy.py)
- [critics/dqn_critic.py](cs285/critics/dqn_critic.py)

You will then need to implement code in the following files:

For RND + CQL:
- [exploration/rnd_model.py](cs285/exploration/rnd_model.py)
- [agents/explore_or_exploit_agent.py](cs285/agents/explore_or_exploit_agent.py)
- [critics/cql_critic.py](cs285/critics/cql_critic.py)

For AWAC: 
- [agents/awac_agent.py](cs285/agents/awac_agent.py)

For IQL:
- [agents/iql_agent.py](cs285/agents/iql_agent.py)
- [critics/iql_critic.py](cs285/critics/iql_critic.py)


The relevant sections are marked with `TODO`.

You may also want to look through [scripts/run_hw5_expl.py](cs285/scripts/run_hw5_expl.py) (if running locally) or [run_hw5_expl.ipynb](run_hw5_expl.ipynb) (if running on Colab), though you will not need to edit this files beyond changing runtime arguments in the Colab notebook.

See the [assignment PDF](hw5.pdf) for more details on what files to edit.

For this particular assignment, you will need to install networkx==2.5
