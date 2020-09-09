import torch
import torch.nn as nn
import torch.nn.functional as F


#Dueling Network Architecture
class Network(nn.Module):
    def __init__(self,state_size,action_size,seed, fc1_units = 64,fc2_units = 64,adv_fc1_units=64,val_fc1_units=64):
        super(Network,self).__init__()
        self.action_size = action_size
        self.state_size = state_size
        self.seed = seed
        self.fc1 = nn.Linear(state_size,fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.adv_fc1 = nn.Linear(fc2_units,adv_fc1_units)
        self.val_fc1 = nn.Linear(fc2_units, val_fc1_units)
        self.val_fc2 = nn.Linear(val_fc1_units,1)
        self.adv_fc2 = nn.Linear(adv_fc1_units, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        val = F.relu(self.val_fc1(x))
        adv = F.relu(self.adv_fc1(x))
        val = F.relu(self.val_fc2(val))
        adv = F.relu(self.adv_fc2(adv))
        y = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0),self.action_size)
        return y
       