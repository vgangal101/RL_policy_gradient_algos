import jsonargparse
import torch 
import matplotlib.pyplot as plt 
import gym 
import numpy as np
from networks import BasicMLP
from policy import Policy
from state_value_function import StateValueFunction
from torch.nn import functional as F 

from ppo_utils import TrajectoryDataset, experience



def get_args():
   
    parser = jsonargparse.ArgumentParser()
    parser.add_argument('--config',action=jsonargparse.ActionConfigFile)
    parser.add_argument('--network_name',default='BasicMLP',type=str)
    parser.add_argument('--env_name',default='CartPole-v1',type=str)
    parser.add_argument('--num_episodes',default=1000,type=int)
    parser.add_argument('--num_timesteps',default=10000,type=int)
    parser.add_argument('--gamma',type=float)
    parser.add_argument('--policy_step_size',type=float)
    parser.add_argument('--state_value_step_size',type=float)
    parser.add_argument('--optimizer',default='Adam',type=str)

    return parser.parse_args()



def train(params):
    """
    Train on cartpole or classic control tasks 
    """

    # initialize environment. using args 
    env_name = params.env_name
    env = gym.make(env_name)

    num_episodes = params.num_episodes
    num_timesteps = params.num_timesteps 
    gamma = params.gamma # make it a default of 0.99 
    step_size = params.step_size # is learning rate 


    policy_config = dict(network_name=params.network_name,obs_space=env.observation_space.shape,action_space=env.action_space.n)
    policy = Policy(policy_config)

    if params.optimizer == 'Adam':
        optimizer = torch.optim.Adam(policy.network.parameters(),lr=step_size)
    elif params.optimizer == 'SGD':
        optimizer = torch.optim.SGD(policy.network.parameters(),lr=step_size)

    
    trajectory_dataset = TrajectoryDataset()
    train_rewards_perf = []
    for i in range(num_episodes):
        obs = env.reset()
        #log_prob_action_tracker = []
        rewards_tracker = []
        train_rewards_episode = 0

        for ts in range(num_timesteps):
            action, _ = policy.select_action(obs)
            #log_prob_action_tracker.append(log_prob_action)
            next_obs, reward, done, info = env.step(action)
            sample = experience(obs,action,reward,next_obs,done)
            trajectory_dataset.store(sample)
            obs = next_obs
            
            rewards_tracker.append(reward)
            train_rewards_episode += reward
            
            if done: 
                break
        train_rewards_perf.append(train_rewards_episode)
        
        update_policy(policy, optimizer, log_prob_action_tracker,rewards_tracker,gamma)

        print(f'episode={i}  episode_reward={train_rewards_episode}')

    return train_rewards_perf


# FIX THE METHOD SIGNATURE 
def update_policy(policy,state_val_func,policy_optimizer,state_val_func_optimizer,trajectory_dataset,gamma):

    # compute rewards to go ( same way the discounted rewards are computed in REINFORCE)
    discounted_returns = []
    for i in range(len(trajectory_dataset)):
        Gt = 0
        power = 0 
        for index in range(i,len(trajectory_dataset)):
            Gt += trajectory_dataset[index].reward * gamma ** power
            power += 1 
        discounted_returns.append(Gt)

    discounted_returns = torch.tensor(discounted_returns)

    # compute advantage estimates



    
    return 
