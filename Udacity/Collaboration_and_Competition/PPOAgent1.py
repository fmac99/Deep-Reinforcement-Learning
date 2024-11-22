import numpy as np
import random
import copy
import torch.nn as nn
from collections import namedtuple, deque
from torch.autograd import Variable
from PPOModel1 import Policy, Value
from torch.distributions import Normal
import torch
import torch.nn.functional as F
import torch.optim as optim
            
LR_POLICY = 3e-4         # learning rate of the actor 
LR_VALUE = 1e-6      # learning rate of the critic
WEIGHT_DECAY = 0.0       # L2 weight decay

device = torch.device("xpu:0" if torch.cuda.is_available() else "cpu")


class PPO_Agent():
    def __init__(self,state_size, action_size, random_seed, n_actors, batch_size,mini_batch,gamma,lambduh,tmax,opt_epochs,epsilon):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.n_actors = n_actors
        self.batch_size=batch_size
        self.mini_batch = mini_batch
        self.gamma = gamma
        self.lambduh = lambduh
        self.tmax = tmax
        self.opt_epochs = opt_epochs
        self.epsilon = epsilon
        self.csf_beta = .02
        
        self.policy_net = Policy(action_size,state_size, random_seed).to(device)
        self.policy_optim = optim.Adam(self.policy_net.parameters(),lr = LR_POLICY,eps=1e-4,amsgrad=True)
       
        self.value_target = Value(state_size, random_seed).to(device)
        self.value_optim = optim.Adam(self.value_target.parameters(),lr=LR_VALUE)
        self.memory = ReplayBuffer(state_size,action_size, random_seed)
        
        
    def ClippedSurrogateFunction(self,states,probs, advantages,actions,epsilon,Minb):
        '''
        ClipSF = L_clip + c_1*L_VF + c_2*S
        L_clip = torch.min(prob_ratio * Adv, torch.clamp(prob_ratio,1-epsilon,1+epsilon)*Adv)
        L_VF = MSE_Value function --> Value_Loss
        S = Entropy
        c_1,c_2 = constants
        '''
        states = torch.tensor(states,dtype=torch.float)
        dist = self.policy_net.forward(states)
        new_probs = dist.log_prob(actions).sum(dim=2)
        entropy = dist.entropy()
        beta = self.csf_beta
        prob_ratio = (new_probs-probs).exp()
        S1 =advantages.reshape(Minb,self.n_actors)*prob_ratio
        S2 = advantages.reshape(Minb,self.n_actors)*torch.clamp(prob_ratio,1-epsilon,1+epsilon)
        L_clip = torch.min(S1,S2)
        Clipped_Surrogate_Function = -(L_clip + beta * entropy.sum(dim=2)).mean()
    
        return Clipped_Surrogate_Function
   
        
        
   
    def Advantage_Estimation(self,values, rewards,gamma,lambduh, tmax):
        vals=values
        vals.append(torch.zeros(1,requires_grad=True))
        delta_list=[]
        lg = (gamma*lambduh)**np.arange(len(rewards))
        l = lg[::-1]
        ten=torch.tensor(l.copy(),dtype=torch.float).reshape(len(rewards),1)
        for t in range(len(rewards)):
            k = t+1
            delta= -vals[t]+vals[k]*gamma+torch.tensor(rewards[t],dtype=torch.float).reshape(2,1)
            delta_list.append(delta)

        delta_list = delta_list[::-1]

        tensor_list=[]
        for t in range(len(delta_list)):
            tensor_list.append(delta_list[t].squeeze())

        deltas=torch.cat(tensor_list).reshape(tmax,self.n_actors)
        Advantage_List = []
        for x in range(len(deltas)):
            Advantage_List.append((deltas[0:len(deltas)-x]*ten[x:len(deltas)]).sum(dim=0))

        advantages=torch.cat(Advantage_List)
        advantages = advantages.reshape(tmax,2)

       

        Normalized_Advan=(advantages -advantages.mean())/(advantages.std()+1e-10)
        return Normalized_Advan


    def Value_Estimation(self,rewards, gamma):
        rewards_list = []
        disco = gamma**np.arange(len(rewards))
        disco = disco[::-1]
        disco = disco.reshape(len(rewards),1)
        #dt = torch.tensor(disco.copy(),dtype=torch.float).reshape(len(rewards),1)
        rewards = rewards[::-1]
        for r in range(len(rewards)):
            rewards_list.append((rewards[0:len(rewards)-r]*disco[r:len(rewards)]).sum(axis=0))


        rewds = torch.tensor(rewards_list,dtype=torch.float)
        return rewds

    def Learn(self,states,v_states,value_estimate,probs,advs,actions,epsilon,Minb):

        value_target= self.value_target.forward(torch.tensor(v_states,dtype=torch.float)).reshape(Minb,self.n_actors)
        
        value_estimate = value_estimate.reshape(Minb,self.n_actors)
        advs = advs.reshape(Minb,self.n_actors)

        Value_Loss = .5* F.mse_loss(value_target, value_estimate)    
        Value_Loss = Variable(Value_Loss,requires_grad=True)
        self.value_optim.zero_grad()
        Value_Loss.backward()
        self.value_optim.step()
        Policy_Loss=self.ClippedSurrogateFunction(states,probs, advs,actions,epsilon,Minb)
        self.policy_optim.zero_grad()
        Policy_Loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(),.75)
        self.policy_optim.step()
        
        
    def MakeMinibatch(self,states,v_state,value_est,probs,advs,actions,mini_batch):
        batch_dict = {}
        rand_rows= np.random.randint(len(states), size=mini_batch)
        batch_dict = {
            
            "states":torch.from_numpy(np.vstack([states[r] for r in rand_rows])).reshape(mini_batch,self.n_actors,self.state_size),
            "v_states":torch.from_numpy(np.vstack([states[r] for r in rand_rows])).reshape(mini_batch,self.state_size*2),
            "value_est":torch.from_numpy(np.vstack([value_est[r].detach().numpy() for r in rand_rows])),
            "probs":torch.from_numpy(np.vstack([probs[r].detach().numpy() for r in rand_rows])).reshape(mini_batch,self.n_actors),
            "advs":torch.from_numpy(np.vstack([advs[r] for r in rand_rows])),
            "actions":torch.from_numpy(np.vstack([actions[r] for r in rand_rows])).reshape(mini_batch,self.n_actors,self.action_size)
        }
            
        return batch_dict
        
                                                              
    def step(self, states,v_states, actions,probabilities,values, rewards, dones):
    # Save experience / reward
        self.memory.add(states,v_states, actions,probabilities,values, rewards, dones)

    # Learn, if enough samples are available in memory
        if self.memory.mem_size == self.batch_size:
            value_est = self.Value_Estimation(self.memory.reward_list,self.gamma)
            advantages = self.Advantage_Estimation(self.memory.value_list,self.memory.reward_list,\
                                                   self.gamma,self.lambduh,self.tmax)
            advantages = advantages.detach().numpy()
            for _ in range(self.opt_epochs):
                for _ in range(self.batch_size//self.mini_batch):
                    batch=self.MakeMinibatch(self.memory.state_list,self.memory.v_states_list,value_est,\
                                       self.memory.probability_list,advantages,self.memory.action_list,self.mini_batch)
                    self.Learn(batch.get("states"),batch.get("v_states"),batch.get("value_est")\
                            ,batch.get("probs"),batch.get("advs"),batch.get("actions"),self.epsilon,self.mini_batch)
                    batch.clear()
            self.memory.reset_mem()
                 
             
             
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self,state_size, action_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.state_size = state_size
        self.action_size = action_size
        self.state_list = []
        self.v_states_list =[]
        self.action_list = []
        self.probability_list = []
        self.value_list = []
        self.reward_list = []
        self.dones_list = []
        self.seed = random.seed(seed)
        self.mem_size =0
    
    def add(self, states,v_states, actions,probabilities,values, rewards, dones):
        """Add a new experience to memory."""
        self.state_list.append(states)
        self.v_states_list.append(v_states)
        self.action_list.append(actions)
        self.probability_list.append(probabilities)
        self.value_list.append(values)
        self.reward_list.append(rewards)
        self.dones_list.append(dones)
        self.mem_size +=1
    

    def reset_mem(self):
        self.state_list.clear()
        self.v_states_list.clear()
        self.action_list.clear()
        self.probability_list.clear()
        self.value_list.clear()
        self.reward_list.clear()
        self.dones_list.clear()
        self.mem_size =0
    
                                                             