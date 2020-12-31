
# Report Udacity Project 3 - Colaboration and Competition


#### Project Summary

Much of this report will be essentially the same as the report for continous control as I used the same algorithm. In this section I would like to summarize the project and point out the major differences to the other project. This enviroment was much harder than the continous control environment because the length of episodes can be so different. Since episodes end when a player hits the ball out there are instances where there is only one timestep in an episode. That makes it hard to learn with ppo in the same way as it was used in an environment with a set number of timesteps/episode. So for this environment I added a replay buffer for the PPO algorithm. This allowed me to collect enough samples to calculate advantages and value estimates. The next major difference is I used a shared value net between the agents. This allows for the value function of each agent to consider the state of the other agent. Since in a real world tennis match a player would be looking at their opponents position to determine a set of possible moves I think thought this made sense. It is also what is done in implementations of MADDPG. 

The other major hurdle in this environment was stability. I would have runs where the score would get as high as .47 and then deteriorate all the way done to the 20s and never return. Since I could see that the agents were learning something I figured I needed to do something for that. So I tweaked the learning rates on the my neural nets some and found that changing the value fuction learning rate helped tremendously. I also changed the epsilon of the Adam optimizer to try to increase numerical stability. Finally I switched the amsgrad to True on the Adam optimizer. The point of amsgrad is to try to prevent convergence on a suboptimial solution. I noticed that my agents would hover around certain values and go up and down as more episodes went on. All of these changes coupled with letting it run for a large amount of very fast episodes got me to the finish line.


#### PPO Algorithm

For this project I implemented the PPO Algorithm with an A2C network architechture. I choose this because of it is simplicity and effectiveness based on benchmarks. This algorithm uses stochastic gradient ascent with a custom loss function to improve learning.  The nice thing about this Unity enviroment is that it handles all of the multi env/agent creation so there is no need to create custom multiprocessing code.

The crux of this success with this algorithm is making sure the PPO Loss funcition is properly written. The loss function used was the Clipped Surrogate Loss function from the Proximal Policy Optimization Paper referenced in the Readme. The function is as follows:

        Clipped_Surrogate_Funtion = L_clip + c_1*L_VF + c_2*S
        L_clip = min(prob_ratio * Advantage, torch.clamp(prob_ratio,1-epsilon,1+epsilon)*Advantage) (policy)
        L_VF = Mean Squared Error of the Value head or Net(depending on your implementation) --> Value_Loss (also know as critic)
        S = Entropy
        c_1,c_2 = constants, c_1 in this case was .5 and c_2 is also know as beta and was .02 for my implementation.
        epsilon = .01 in my implementation
        prob ratio = new_probabalities/old_probabilities
        Advantage function= GAE 
        
        
Since I used a seperate neural network for the policy and the value functions I had a separate update step and two seperate loss functions. You can combine the above and should if you a shared a network for the policy and the value functions. Then only a singular update step would be required. The L_Clip + c_2*S portion should be negated when using an optimizer such as SGD or Adam so that the algorithm uses Ascent vs Descent. Advantages are estimated using the GAE estimation function(explained below) and the Value Estimates are the discounted rewards for each state. 

#### Neural Network setup

I used and A2C setup for this with initalized with contant values and an orthogonal matrix for both the Policy and Value nets.

##### Policy(Actor)

The neural network for the policy(actor) outputs the mean and standard deviation for the action of the agents and then those are fed into the normal distribution. From there the distribution is sampled for actions, log probabilities and entropy. With pytorch's Normal function all of this is nicely handled and I would highly recommend using that as opposed to writing it yourself. I also made sure to clip those outputs between -1 and 1 in the event they were greater or less than those values. The network architecture is as follows:

Input_Size: state_size=24/agent

Hidden_Units: 22 unit fully connected linear layer

Output_Size: action_size- 2/agent

Total 3 fully connected Linear layers

All using a tanh activation function

A separate paramater is used for the standard deviation which is initalized as 0 for each agent.

##### Value(Critic)
The neural network for the value(critic) outputs a single value for the state value function. This is a value assigned to the states which measures how good a state is. This time I used something similar to what is done in MADDPG implementations and made a shared value net between both agents.  

Input_Size: state_size=48(both agents states in one tensor

Hidden_Units: 46 unit fully connected linear layer

Output_Size: value- 1 value output/agent

Total 3 fully connected Linear layers for each agent with one shared between each

all using a relu activation function except for the output layer

#### Advantage Estimation 

I would reference the Advantage Estimation paper for more details( in readme) but the function itself is as follows(both snippets are from that paper,https://arxiv.org/pdf/1506.02438.pdf):
#### Delta Function:
<p>
  <img src= "https://github.com/fmac99/Deep-Reinforcement-Learning/blob/master/DeltaPic.png">
</p>
<p>
<img src ="https://github.com/fmac99/Deep-Reinforcement-Learning/blob/master/GAEPic.png">
</p>
GAE is what I used for my advantage and I normalized them across all agents and advantages. 


#### Batching, Time Horizons, and Optimization epochs

Many implementations used what they called a rollout. That references the the horizon for n-step bootsrapping. I used the term batch size and made it equivalent to the time horizon I called tmax. I selected 1000 for tmax in this instance because it worked in the last project. I also had a minibatch for learning steps that was 100 time steps in length. That creates 10 learning batches. I used an optimization epoch number of 6. I found that higher numbers got the agent stuck at values and it didn't learn after a while. 

#### Reward Plotting

Below is the plot of rewards. It took my agent 21292 episodes to get 0.5+ for 100 episodes.

<p>
<img src='https://github.com/fmac99/Deep-Reinforcement-Learning/blob/master/Udacity/Collaboration_and_Competition/CCSolution.png'>
</p>



#### Hyperparameters

GAMMA = .99
LAMBDA = .96
Epsilon = 0.2
Beta = .02
TMAX=1000
MINI_BATCH = 100
BATCH_SIZE  = 1000
iterations=6
Episodes =100000
LR_Policy = 3e-4
LR_Value = 1e-6
Adam epsilon for the policy net = eps=1e-4
Adam amsgrad=True

#### Future Improvements

One thing I would like to try for this project is using recurrent neural networks to see if it would help at all. I noticed that many times there would be a really good score on an espisode and then a terrible one the next. I wonder if the memory properties of recurrent neural nets would help with catpuring the most important features of learning to prevent that stark contrast in performance.
