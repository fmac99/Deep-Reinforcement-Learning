# Tennis Env Info and Task

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
 
This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.



# Getting Started with this enviroment and problem

For those of you interested in doing this work or running this code yourself you will need the following things downloaded to enable you to work in the environment. Note these instructions are based on those given in Udacity's Deep Reinforcement Learning Nanodegree. Detailed instructions can be found to setup the python enviroment @ this link

DRLND ENV -[https://github.com/udacity/deep-reinforcement-learning]
The above link will show you in detail how to setup your python virtual environment, tell the dependencies needed for running the code and how to download clone any of the need repositiories/code that will be requried. For those of you how know how to make a virtual enviroment on their cpu and use pip to download python repositories here is a list of dependencies needed in your venv:

# Dependencies
Python 3.6 or greater

Pytorch 0.4.0

Open AI Gym -0.17.2


That will get you started with the python enviroment. Here is a link on installing the project enviroment for the agent to work in. 

Unity Enviroment for Continous Control Project-[https://github.com/fmac99/deep-reinforcement-learning-1/tree/master/p3_collab-compet]

I was able to solve this on cpu 









# References For this problem:

Udacity and code/teaching provided in the deep reinforcement nanodegree program

 "High-Dimensional Continuous Control Using Generalized Advantage Estimation" ,John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel-[https://arxiv.org/abs/1506.02438]
 
"Proximal Policy Optimization Algorithms" John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov- [https://arxiv.org/abs/1707.06347]

"On-Policy Trust Region Policy Optimisation with Replay Buffers" Dmitry Kangin, Nicolas Pugeault -[https://arxiv.org/abs/1901.06212]


"Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel, Igor Mordatch-[https://arxiv.org/abs/1706.02275]

