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
LAMBDA = 0.96
GAMMA = 0.99           # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_POLICY = 3e-4         # learning rate of the actor 
LR_VALUE = 1e-4       # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPO_Agent():
    def __init__(self,state_size, action_size, random_seed, n_actors):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.n_actors = n_actors
        
        self.policy_net_new = Policy(action_size,state_size, random_seed).to(device)
        self.policy_net_old = Policy(action_size,state_size, random_seed).to(device)
        self.policy_optim = optim.Adam(self.policy_net_new.parameters(),lr = LR_POLICY)
       
        self.value_local = Value(state_size,random_seed).to(device)
        self.value_target = Value(state_size, random_seed).to(device)
        self.value_optim = optim.Adam(self.value_target.parameters(),lr=LR_VALUE)
        
    def ClippedSurrogateFunction(self,states,probs, advantages,epsilon,Minb):
        '''
        ClipSF = L_clip + c_1*L_VF + c_2*S
        L_clip = torch.min(prob_ratio * Adv, torch.clamp(prob_ratio,1-epsilon,1+epsilon)*Adv)
        L_VF = MSE_Value function --> Value_Loss
        S = Entropy
        c_1,c_2 = constants
        '''
        states = torch.tensor(states,dtype=torch.float)
        probs = probs.view([Minb,20,4])
        dist = self.policy_net_new(states)
        actions = dist.sample()
        new_probs = dist.log_prob(actions).view([Minb,20,4])
        entropy = dist.entropy()
        beta = .02
        prob_ratio = (new_probs-probs).exp() 
#        prob_ratio = prob_ratio.mean(dim=2)
        S1 =(advantages.reshape(Minb,20,1)*prob_ratio).mean()
        S2 = (advantages.reshape(Minb,20,1)*torch.clamp(prob_ratio,1-epsilon,1+epsilon)).mean()
        L_clip = torch.min(S1,S2)
        Clipped_Surrogate_Function = -L_clip  - beta * entropy.mean()
    
        return  Clipped_Surrogate_Function
   
        
   
    def NetworkUpdate(self,local_model,target_model):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)
        
        
    def Advantage_Estimation(self,values, rewards,gamma,lambduh, tmax):
        values.append(torch.zeros(20,1))
        delta_list=[]
        lg = (gamma*lambduh)**np.arange(len(rewards))
        l = lg[::-1]
        ten=torch.tensor(l.copy(),dtype=torch.float).reshape(tmax,1)
        for t in range(tmax):
            k = t+1
            delta= -values[t]+values[k]*gamma+torch.tensor(rewards[t],dtype=torch.float).reshape(20,1)
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
 
        Normalized_Advan=(advantages -advantages.mean(dim=0))/(advantages.std(dim=0)+1e-10)
        return Normalized_Advan


    
                                                              
    def Value_Estimation(self,rewards, gamma):
        value_list = []
        disco = gamma**np.arange(len(rewards))[:,np.newaxis]
        for r in range(len(rewards)):
            value_list.append((rewards[0:len(rewards)-r]*disco[r:len(rewards)]).sum(axis=0))

        return torch.tensor(value_list,dtype=torch.float)

    def Learn(self,states,value_targets,value_estimate,probs,advs,epsilon,Minb):
        
        value_target =value_targets.reshape(Minb,20)
        value_estimate = value_estimate.reshape(Minb,20)
        advs = advs.reshape(Minb,20)
        value_target=Variable(value_target+advs, requires_grad = True)
        value_estimate=Variable(value_estimate, requires_grad = True)
        Value_Loss = F.mse_loss(value_target, value_estimate)    
        
        self.value_optim.zero_grad()
        Value_Loss.backward()
        self.value_optim.step()
        
        Policy_Loss=Variable(self.ClippedSurrogateFunction(states,probs, advs,epsilon,Minb),requires_grad=True)
        self.policy_optim.zero_grad()
        Policy_Loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net_new.parameters(),.5)
        self.policy_optim.step()
        
        
    def MakeMinibatch(self,states,values,value_est,probs,advs,tmax,mini_batch):
        batch_dict = {}
        rand_rows= np.random.randint(tmax-1, size=mini_batch)
        batch_dict = {
            
            "states":torch.from_numpy(np.vstack([states[r] for r in rand_rows])),
            "values":torch.from_numpy(np.vstack([values[r].detach().numpy() for r in rand_rows])),
            "value_est":torch.from_numpy(np.vstack([value_est[r] for r in rand_rows])),
            "probs":torch.from_numpy(np.vstack([probs[r].detach().numpy() for r in rand_rows])),
            "advs":torch.from_numpy(np.vstack([advs[r] for r in rand_rows])),
#            "returns":torch.from_numpy(np.vstack([returns[r] for r in rand_rows]))
        }
            
        return batch_dict
        
                                                              
        
                                                              
