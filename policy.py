import networks 
import torch
import numpy as np

class Policy():
    def __init__(self,config):
        self.num_actions = config['action_space']
        if config['network_name'] == 'BasicMLP': 
            self.network = networks.BasicMLP(config['obs_space'],config['action_space'])
        else: 
            return ValueError("Invalid network name")
        
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.network(state)
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob


