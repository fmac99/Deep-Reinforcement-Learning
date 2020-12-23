# Reacher Env Info and Task


In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

There are two flavors of this env for the Udacity versions of this env. One with a single arm and one with 20 arms. A successful implementation will achieve an average score of 30 over 100 episodes. In the 20 arm version it needs to be an average of 30 across all agents for 100 episodes



# Getting Started with this enviroment and problem

For those of you interested in doing this work or running this code yourself you will need the following things downloaded to enable you to work in the environment. Note these instructions are based on those given in Udacity's Deep Reinforcement Learning Nanodegree. Detailed instructions can be found to setup the python enviroment @ this link

DRLND ENV -[https://github.com/udacity/deep-reinforcement-learning]
The above link will show you in detail how to setup your python virtual environment, tell the dependencies needed for running the code and how to download clone any of the need repositiories/code that will be requried. For those of you how know how to make a virtual enviroment on their cpu and use pip to download python repositories here is a list of dependencies needed in your venv:

# Dependencies
Python 3.6 or greater

Pytorch 0.4.0

Open AI Gym -0.17.2


That will get you started with the python enviroment. Here is a link on installing the project enviroment for the agent to work in. 

Unity Enviroment for Continous Control Project- [https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control]

This will get you the banana collection enviroment I used to train the agent in the code. Once you have created your enviroment the only augementation you will need to do to the code in this repository to make it run is to enter in the file path for the Unity enviroment on your cpu.



# PPO Agent To Solve Reacher 20 Arm ENV

For this project I selected the PPO algorithm because of it is simplicity and efficiency for solving continuous control problems. I had several challenges along the way but here are the key points I would make sure you have to get any PPO implementation to work:

1. In a continuous env use a sampling distribution to get actions and probabilities. It just makes it way easier to have the Policy Net output the mean and std of the Normal distribution.

2. Make sure you calculate the GAE and Value estimation properly. The paper for these calculations are provided below. These have to be correct for your implementation to work. My implementation was a little different than others but i achieves the goal. 

3. Use the sampled actions from the Old Policy when running the learn step. One thing I did wrong was I used new actions when running my Clipped Surrogate Function. This caused my agent to just act randomly and not learn anything. Use the actions collected when gathering trajectories and those are fed into the learing steps policy distribution to get log probablities. This will give the agent the correct probability ratio.

4. Use weight initialization on your NNs. It just greatly helped the performance of my learning process.

5.If you get stuck use other successful implementations to help you check yours and see what the differences are. 










# References For this problem:

Udacity and code/teaching provided in the deep reinforcement nanodegree program

 "High-Dimensional Continuous Control Using Generalized Advantage Estimation" ,John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel-[https://arxiv.org/abs/1506.02438]
 
"Proximal Policy Optimization Algorithms" John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov- [https://arxiv.org/abs/1707.06347]

Implementations that helped me fix errors in mine:
Shangtong Zhang-[https://github.com/ShangtongZhang/DeepRL]
Simon Birrell- [https://github.com/SimonBirrell/ppo-continuous-control]
