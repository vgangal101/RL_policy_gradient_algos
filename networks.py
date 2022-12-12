from torch import nn
import torch.nn.functional as F 

class BasicMLP(nn.Module):
    def __init__(self,obs_dims,action_dims):
        super().__init__()
        self.obs_dims = obs_dims[0]
        self.action_dims = action_dims

        # layers 
        self.linear1 = nn.Linear(self.obs_dims,64)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(64,self.action_dims)
    
    def forward(self,x):
        x = self.linear1(x)
        x = self.linear2(x)
        return F.softmax(x,dim=-1)

