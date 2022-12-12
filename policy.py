import networks 
import torch

class Policy():
    def __init__(self,config):
        if config['network_name'] == 'BasicMLP': 
            self.network = networks.BasicMLP(config['obs_space'],config['action_space'])
        else: 
            return ValueError("Invalid network name")
        
    def select_action(self,obs):
        obs = torch.from_numpy(obs)
        with torch.no_grad():
            action_probs = self.network(obs)
            action_distribution = torch.distributions.Categorical(action_probs)
            action = action_distribution.sample()
        return action.item()

    def compute_log_prob_action(self,obs,action):
        obs = torch.from_numpy(obs)
        action_probs = self.network(obs)
        action_distribution = torch.distributions.Categorical(action_probs)
        log_action_prob = action_distribution.log_prob(torch.tensor([action]))
        return log_action_prob 
    



