import networks 
import torch
import numpy as np

class Policy():
    def __init__(self,config):
        if config['network_name'] == 'BasicMLP': 
            self.network = networks.BasicMLP(config['obs_space'],config['action_space'])
        else: 
            raise ValueError("Invalid network name")
        
    def select_action(self,obs):
        obs = torch.from_numpy(obs)
        action_probs = self.network(obs)
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample()
        log_prob_action = action_distribution.log_prob(action)
        return action.item(), log_prob_action

    def compute_log_prob_action(self,obs,action):
        if isinstance(obs,np.ndarray): 
            obs = torch.from_numpy(obs)
        action_probs = self.network(obs)
        action_distribution = torch.distributions.Categorical(action_probs)
        log_prob_action = action_distribution.log_prob(action)
        return log_prob_action
    



