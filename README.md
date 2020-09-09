# Deep-Reinforcement-Learning/Reinforcement-Learning

This reposititory is a collection of work in Deep Reinforcement Leanring and regular Reinforcement Learning. 


# Udacity Program Work



# Project 1 Navigation:

In this project I was tasked with creating a Deep Q Network that solves a Unity environment where an agent is collects bananas. In the enviroment the agent can select the following actions:
- `0` - walk forward 
- `1` - walk backward
- `2` - turn left
- `3` - turn right

The state space had a size of 37 spaces. The agent recieved a reward of +1 for collecting yellow bananas and -1 for collecting blue bananas. To solve the enviroment I had to get the agent to get a score of at least 13.0 for 100 episodes and try to do so in less than 1800 episodes. To do so you can employ several methods to solve this. Of course you can use the vanilla DQN model. When I did the project I took the DQN model from one of the other projects done in the course and changed some of the inputs so it would work with the environment I was using. The performed very well, each run it seemed to solve the enviroment in about 450-490 episodes. 

With that in mind the to improve the model there were several options to do so. The options in this case are Double Q Learning, Dueling Network Architecture and prioritized experience replay. There are other ways to make improvements but these are the 3 I expored as presented in the project rubric. The coolest thing about these methods is the approach different parts of the problem so the all of them can be combined for maximal benefit. I tried them all but settled on my submission being just a Dueling Network change.

# Dueling Network Architecture

In Deep Reinforement Learning the Dueling Network Architecture refers to a neural network architecture that has two streams instead of just a single stream to the q value. This is arrived at by spliting out the value and advantage functions into seperate streams. 

