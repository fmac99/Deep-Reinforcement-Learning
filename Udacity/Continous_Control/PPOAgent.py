import numpy as np
import random
import copy
import torch.nn as nn
from collections import namedtuple, deque
from torch.autograd import Variable
from PPOModel import Policy, Value
from torch.distributions import Normal
import torch
import torch.nn.functional as F
import torch.optim as optim
#LAMBDA = 0.96           # GAE Lambda
#GAMMA = 0.99           # discount factor
LR_POLICY = 3e-4         # learning rate of the actor 
LR_VALUE = 3e-4      # learning rate of the critic
WEIGHT_DECAY = 0.0       # L2 weight decay
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPO_Agent():
    def __init__(self,state_size, action_size, random_seed, n_actors):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.n_actors = n_actors
        
        self.policy_net = Policy(action_size,state_size, random_seed).to(device)
        self.policy_optim = optim.Adam(self.policy_net.parameters(),lr = LR_POLICY,eps=1e-5)
       
        self.value_target = Value(state_size, random_seed).to(device)
        self.value_optim = optim.Adam(self.value_target.parameters(),lr=LR_VALUE)
        
      
    # Clipped Surrogate Function Loss Function for Policy    
    def ClippedSurrogateFunction(self,states,probs, advantages,actions,epsilon,Minb):
       
        #ClipSF = L_clip + c_1*L_VF + c_2*S
        #L_clip = torch.min(prob_ratio * Adv, torch.clamp(prob_ratio,1-epsilon,1+epsilon)*Adv)
        #L_VF = MSE_Value function --> Value_Loss-done seperatly
        #S = Entropy
        #c_1,c_2 = constants- c_1 =.5, c_2= beta=.02
        
        states = torch.tensor(states,dtype=torch.float)
        dist = self.policy_net.forward(states)
        new_probs = dist.log_prob(actions).sum(dim=2)
        entropy = dist.entropy()
        beta = .02
        prob_ratio = (new_probs-probs).exp()
        S1 =advantages.reshape(Minb,self.n_actors,)*prob_ratio
        S2 = advantages.reshape(Minb,self.n_actors)*torch.clamp(prob_ratio,1-epsilon,1+epsilon)
        L_clip = torch.min(S1,S2)
        Clipped_Surrogate_Function = -(L_clip + beta * entropy.sum(dim=2)).mean()
    
        return Clipped_Surrogate_Function
   
        
   
  
        
    # Calculate Advantage Estimates, Using GAE formula from Paper Referenced in ReadME
    def Advantage_Estimation(self,values, rewards,gamma,lambduh, tmax):
        vals=values
        vals.append(torch.zeros(20,1,requires_grad=True))
        delta_list=[]
        lg = (gamma*lambduh)**np.arange(len(rewards))
        l = lg[::-1]
        ten=torch.tensor(l.copy(),dtype=torch.float).reshape(tmax,1)
        for t in range(tmax):
            k = t+1
            delta= -vals[t]+vals[k]*gamma+torch.tensor(rewards[t],dtype=torch.float).reshape(20,1)
            delta_list.append(delta)

        delta_list = delta_list[::-1]

        tensor_list=[]
        for t in range(len(delta_list)):
            tensor_list.append(delta_list[t].squeeze())

        deltas=torch.cat(tensor_list).reshape(tmax,20)
        Advantage_List = []
        for x in range(len(deltas)):
            Advantage_List.append((deltas[0:len(deltas)-x]*ten[x:len(deltas)]).sum(dim=0))

        advantages=torch.cat(Advantage_List)
        advantages = advantages.reshape(tmax,20)

        Normalized_Advan=(advantages -advantages.mean())/(advantages.std()+1e-10)
        return Normalized_Advan

    #Calculate Discounted Rewards as an estimate for the Value Function
    def Value_Estimation(self,rewards, gamma,values):
        vals = []
        for v in range(len(values)):
            vals.append(values[v].squeeze())
        vals =vals[::-1]
        vals = torch.cat(vals).reshape(len(rewards),20)
        rewards_list = []
        disco = gamma**np.arange(len(rewards))
        disco = disco[::-1]
        disco = disco.reshape(len(rewards),1)
        dt = torch.tensor(disco.copy(),dtype=torch.float).reshape(len(rewards),1)
        rewards = rewards[::-1]
        for r in range(len(rewards)):
            rewards_list.append((rewards[0:len(rewards)-r]*disco[r:len(rewards)]).sum(axis=0))

        rewds = torch.tensor(rewards_list,dtype=torch.float)
      
        return rewds
        
    # Learn Step using Clipped Surrogate Function for Policy and MSE Loss for Value
    def Learn(self,states,value_estimate,probs,advs,actions,epsilon,Minb):
        
        value_target =self.value_target.forward(torch.tensor(states,dtype=torch.float)).reshape(Minb,self.n_actors)
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
        
    # Make MiniBatches from Collected Trajectories
    def MakeMinibatch(self,states,value_est,probs,advs,actions,tmax,mini_batch):
        batch_dict = {}
        rand_rows= np.random.randint(tmax, size=mini_batch)
        batch_dict = {
            
            "states":torch.from_numpy(np.vstack([states[r] for r in rand_rows])).reshape(mini_batch,self.n_actors,self.state_size),
            "value_est":torch.from_numpy(np.vstack([value_est[r].detach().numpy() for r in rand_rows])),
            "probs":torch.from_numpy(np.vstack([probs[r].detach().numpy() for r in rand_rows])).reshape(mini_batch,self.n_actors),
            "advs":torch.from_numpy(np.vstack([advs[r] for r in rand_rows])),
            "actions":torch.from_numpy(np.vstack([actions[r] for r in rand_rows])).reshape(mini_batch,self.n_actors,self.action_size)
        }
            
        return batch_dict
        
                                                              
        
                                                              
