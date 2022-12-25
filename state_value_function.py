import networks
import torch

class StateValueFunction():
    def __init__(self,config):
        if config['network_name'] == 'BasicMLP':
            self.network = networks.BasicMLP(config['obs_space'],config['action_space'],config['use_softmax'])
    
    def forward(self,obs):
        obs = torch.from_numpy(obs)
        state_value = self.network(obs)
        return state_value
        

