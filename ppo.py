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
    parser.add_argument('--policy_num_iterations',default=10,type=int)
    parser.add_argument('--state_val_num_iterations',default=10,type=int)
    parser.add_argument('--epsilon',default=0.2,type=float)

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

    policy_num_iterations = params.policy_num_iterations
    state_val_num_iterations = params.state_val_iterations

    policy_config = dict(network_name=params.network_name,obs_space=env.observation_space.shape,action_space=env.action_space.n)
    policy = Policy(policy_config)

    state_value_function_config = dict(network_name=params.network_name,obs_space=env.observation_space.shape,action_space=1,use_softmax=False)
    state_val_func = StateValueFunction(state_value_function_config)

    if params.optimizer == 'Adam':
        policy_optimizer = torch.optim.Adam(policy.network.parameters(),lr=params.policy_step_size)
        state_value_optimizer = torch.optim.Adam(state_val_func.network.parameters(),lr=params.state_value_step_size)
    elif params.optimizer == 'SGD':
        policy_optimizer = torch.optim.SGD(policy.network.parameters(),lr=params.policy_step_size)
        state_value_optimizer = torch.optim.SGD(state_val_func.network.parameters(),lr=-params.state_value_step_size)


    trajectory_dataset = TrajectoryDataset()
    train_rewards_perf = []
    for i in range(num_episodes):
        obs = env.reset()
        #log_prob_action_tracker = []
        rewards_tracker = []
        train_rewards_episode = 0

        for ts in range(num_timesteps):
            action, log_prob_action = policy.select_action(obs)
            #log_prob_action_tracker.append(log_prob_action)
            next_obs, reward, done, info = env.step(action)
            sample = experience(obs,action,reward,next_obs,done,log_prob_action)
            trajectory_dataset.store(sample)
            obs = next_obs
            
            rewards_tracker.append(reward)
            train_rewards_episode += reward
            
            if done: 
                break
        train_rewards_perf.append(train_rewards_episode)
        
        
        update(policy,state_val_func,policy_optimizer,state_value_optimizer,trajectory_dataset,gamma) 
        trajectory_dataset.clear()
        print(f'episode={i}  episode_reward={train_rewards_episode}')

    return train_rewards_perf




        

# FIX THE METHOD SIGNATURE 
def update(policy,state_val_func,policy_optimizer,state_val_func_optimizer,trajectory_dataset,gamma,policy_num_iterations,state_num_iterations,epsilon):

    # WRITE OUT WHAT NEEDS TO BE DONE FOR PSEUDOCODE STEPS 6 & 7 

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

    # update policy by updating PPO-clip objective 

    """
    Initial intuition for Steps 6 and 7 suggest that the idea will be very similar to what is done for 
    supervised learning. Keep in mind as you progress
    """
    for iter_num in range(policy_num_iterations):
        log_prob_actions = []
        curr_log_probs = []
        for s_index in range(len(trajectory_dataset)):
            sample = trajectory_dataset[s_index]
            log_prob = policy.compute_log_prob(sample.obs,sample.action)
            curr_log_probs.append(log_prob)
            log_prob_actions.append(sample.log_prob_action)

        # compute advantage estimates
        # compute TD error as advantage estimate
        advs = [] 
        for s_index in range(len(trajectory_dataset)):
            sample = trajectory_dataset[s_index]
            with torch.no_grad():
                adv = sample.r + gamma * state_val_func.forward(sample.next_obs) - state_val_func.forward(sample.obs)
            advs.append(adv)
        advs = torch.stack(advs)
        
        curr_log_probs = torch.stack(curr_log_probs)
        log_prob_actions = torch.stack(log_prob_actions)
        ratio = torch.exp(curr_log_probs - log_prob_actions)
        
        surr1 = ratio * advs
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advs 
        surr_final_value = (torch.min(surr1,surr2)).mean()

        actor_loss = -1 * surr_final_value
        
        policy_optimizer.zero_grad()
        actor_loss.backward()
        policy_optimizer.step()

    
    # update state_value function by MSE loss 
    for iter_num in range(state_num_iterations):
        state_vals = []
        for s_index in range(len(trajectory_dataset)):
            sample = trajectory_dataset[s_index]
            state_val = state_val_func.forward(sample.obs)
            state_vals.append(state_val)
            
        state_vals = torch.stack(state_vals)
        state_val_loss = F.mse_loss(state_val_loss,discounted_returns)

        state_val_func_optimizer.zero_grad()
        state_val_loss.backward()
        state_val_func_optimizer.step()

    return 
