import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer
        

class Policy(nn.Module):
    def __init__(self,action_size,state_size,seed):
        super(Policy,self).__init__()
        self.action_size = action_size
        self.state_size = state_size
        self.seed = seed
        self.fc1 = layer_init(nn.Linear(state_size,30))
        self.fc2 = layer_init(nn.Linear(30,30))
        self.fc3_mean = layer_init(nn.Linear(30,action_size),1e-3)
        self.std = nn.Parameter(torch.zeros(self.action_size))
        
    def forward(self,state):
        x = F.tanh(self.fc1(state))
        x = F.tanh(self.fc2(x))
        mean =F.tanh(self.fc3_mean(x))
        dist = Normal(mean,F.softplus(self.std))
        return dist

class Value(nn.Module):
    def __init__(self,state_size, seed):
        super(Value,self).__init__()
        self.state_size = state_size
        self.seed = seed
        self.fc1 = layer_init(nn.Linear(state_size,30))
        self.fc2 = layer_init(nn.Linear(30,30))
        self.val = layer_init(nn.Linear(30,1),1e-3)

        
    def forward(self,state):
        y = F.relu(self.fc1(state))
        y = F.relu(self.fc2(y))
        value= self.val(y)
        
        return value

      
  