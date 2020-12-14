import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Policy(nn.Module):
    def __init__(self,action_size,state_size,seed,std=0.0):
        super(Policy,self).__init__()
        self.action_size = action_size
        self.state_size = state_size
        self.seed = seed
        self.fc1 = nn.Linear(state_size,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3_mean = nn.Linear(128,action_size)
#        self.fc3_std = nn.Linear(128,action_size)
#        self.std = nn.Parameter(torch.zeros(action_size))
#        self.fc4 = nn.Linear(64,action_size)
#        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.std = torch.tensor([0,0,0,0])
        self.mean = torch.tensor([0,0,0,0])
        self.apply(init_weights)
        self.log_std = nn.Parameter(torch.ones(1,action_size)*std)
    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
#        x = F.relu(self.fc3(x))
        
        self.mean = self.tanh(self.fc3_mean(x))
        self.std  = self.log_std.exp().expand_as(self.mean)
        dist = Normal(self.mean,self.std)
        return dist

class Value(nn.Module):
    def __init__(self,state_size, seed):
        super(Value,self).__init__()
        self.state_size = state_size
        self.seed = seed
        self.fc1 = nn.Linear(state_size,256)
        self.fc2 = nn.Linear(256,128)
        self.val = nn.Linear(128,64)
        self.val2 = nn.Linear(64,1)
        self.reset_parameters()
        
    def forward(self,state):
        y = F.relu(self.fc1(state))
        y = F.relu(self.fc2(y))
        y = F.relu(self.val(y))
        value = self.val2(y)
        return value
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.val.weight.data.uniform_(*hidden_init(self.val))
        self.val2.weight.data.uniform_(-3e-3, 3e-3)
