# Report Udacity Project 2 - Continous Control

#### PPO Algorithm

For this project I implemented the PPO Algorithm with an A2C network architechture. I choose this because of it is simplicity and effectiveness based on benchmarks. This algorithm uses stochastic gradient ascent with a custom loss function to improve learning. PPO also allows for multiple copies of the agent and the enviroment to be solved simultaniously so I used the 20 agent version of this env. The nice thing about this Unity enviroment is that it handles all of the multi env/agent creation so there is no need to create custom multiprocessing code.

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

Input_Size: state_size=33/agent
Hidden_Units: 30 unit fully connected linear layer
Output_Size: action_size- 4/agent
Total 3 fully connected Linear layers
All using a tanh activation function
A separate paramater is used for the standard deviation which is initalized as 0 for each agent.

##### Value(Critic)
The neural network for the value(critic) outputs a single value for the state value function. This is a value assigned to the states which measures how good a state is. In this case it is very fitting since we are looking to be in the "reward region" for the arm. 

Input_Size: state_size=33/agent
Hidden_Units: 30 unit fully connected linear layer
Output_Size: value- 1/agent
Total 3 fully connected Linear layers
all using a relu activation function except for the output layer

#### Advantage Estimation

I would reference the Advantage Estimation paper for more details( in readme) but the function itself is as follows:
GAE= Generalized Advantage Estimation
GAE= Sum((Lambda*Gamma)**l*Delta_V_l+t) 
Lambda= .95 in my implementation, it is a discounting factor that helps reduce variance
Gamma = .99 in my implementation it is the discounting factor for each rewards and deltas which is 
<p>
<img src="https://github.com/fmac99/Deep-Reinforcement-Learning/blob/master/Udacity/Continous_Control/Solution.png">
</p>
