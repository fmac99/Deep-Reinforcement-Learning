# Deep-Reinforcement-Learning/Reinforcement-Learning

This reposititory is a collection of work in Deep Reinforcement Leanring and regular Reinforcement Learning. 


# Udacity Program Work



# Project 1 Navigation:

In this project I was tasked with creating a Deep Q Network that solves a Unity environment where an agent is collects bananas. In the enviroment the agent can select the following actions:
- `0` - walk forward 
- `1` - walk backward
- `2` - turn left
- `3` - turn right

The state space had a size of 37 spaces in a matrix(no conv nets needed). The agent recieved a reward of +1 for collecting yellow bananas and -1 for collecting blue bananas. To solve the enviroment I had to get the agent to get a score of at least 13.0 for 100 episodes and try to do so in less than 1800 episodes.  course You can use the vanilla DQN model. To get started on the project I took code from the DQN model from one of the other mini projects done in the course and changed some of the inputs so it would work with the environment I was using. The performed very well, each run it seemed to solve the enviroment in about 450-490 episodes. Since I had Udacity's code working well and solving the problem I used some of the google papers provided by Udacity to learn other ways to solve a Deep Q Problem more efficiently.

The options in this case are Double Q Learning, Dueling Network Architecture and prioritized experience replay. There are other ways to make improvements but these are the 3 I expored as presented in the project rubric. The coolest thing about these methods is the approach different parts of the problem so the all of them can be combined for maximal benefit. I tried them all but settled on my submission being just a Dueling Network change because I was not getting good results with some of my other implementations.  

# Dueling Network Architecture

In Deep Reinforement Learning the Dueling Network Architecture refers to a neural network architecture that has two streams instead of just a single stream to the q value. This is arrived at by spliting out the value and advantage functions into seperate streams. And then using those values to caclculate a Q value as a linear combination of the value function reduced to a scalar, and the difference advantage function tensor and the mean or max of the advantage function tensor. It turns out that improves performance of network by allowing for a more generalized answer without changing underlying DQN learning algorithm as cited in the paper "Dueling Network Architecures for Deep Reinforecment Learning". I was able to get a roughly 11-13% performance improvement with this method over the vanilla DQN. Both the regular DQN and one with a Dueling Network ran in a reasonable time on CPU.(5-10 minutes).

Hyperparameters used:

BUFFER_SIZE =  int(1e5)

BATCH_SIZE = 32

LR = 5e-4

UPDATE_EVERY = 4

GAMMA = .99

TAU = 1e-3

Epsilon Start = 1.0

Epsilon Decay = .995

Epsilon End = .01

Network Architecture:

2 Fully Connected Layers- 1 input, 1 hidden, 64 units each 

Advantage Stream- 2 fully connected layers, 64 unit inputs, final output action space size

Value Stream- 2 fully connected layers, 64 unit inputs, final output of 1

All use Relu activation

Network Output = Val + Adv - mean(Adv) 

# How To Use the Dueling Implementation
To use these files you simply need download the DuelingDQN.py, DQNAgent.py and the UdacityProject1-Navigation.ipynb file and have them in a venv or repository where the ipynb file can access the other two.If you want to use the Unity environment you will need to download that seperately. You can also use a gym enviroment you will just need to adjust the envirionment related code in the ipynb file to match your enviroment. You can make tweaks to the neural net in the  DuelingDQN.py file by chaning units, adding or changing layers. Since this enviroment was based on a matrix of state representations conv nets were not required. If you were going to do this with the pixels you would need to add conv nets would need to replace the fully connected linear layers before the split. If you want to change the Agent or any of the hyperparmeters those are all located in the DQNAgent.py file. All settings in each file are the ones that worked for me to get the results I have. To train the agent in the unity enviroment(once you have it downloaded) you will simply need to execute the code in the ipynb file.  I have also saved a check point if you want to see it work in this specific unity enviroment. Make sure that train mode is = True when training and False when testing.


# Future Work Ideas
To improve a DQN agent like this one I want work on and explore implenting double dqn and prioritized experience replay as well. Doulbe DQN makes it so the agent does not make over-confident value estimates and prioritized experience replay helps the agent rank experiences and make better calls on which ones to replay more frequently. Both improve performance and can be used synergistically with a Dueling Network Architecture.


# References For this problem:

Udacity and code/teaching provided in the deep reinforcement nanodegree program

"Dueling Network Architecures for Deep Reinforecment Learning" - GoogleDeepMind paper arXiv:1511.06581v3 [cs.LG] April 5 2016- Ziyu Wang, Tom Schual, Matteo Hessel, Hado van Hasselt, Marc Lanctot, Nando de Freitas

"Deep Reinforcement Learning with Double Q Learning" GoogleDeepMind paper arXiv:1509.06461v3 [cs.LG] Dec 8 2015- Hado van Hasselt, Arthur Guez and David Silver

"Prioritized Experience Replay" GoogleDeepMind pater- arXiv:1511.05952v4 [cs.LG] Feb 25 2016 - Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver

dxyang/DQN_Pytorch for ideas about dueling network architecture, check out his github repo!
