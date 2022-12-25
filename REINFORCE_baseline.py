import jsonargparse
import torch 
import matplotlib.pyplot as plt 
import gym 
from collections import namedtuple
import numpy as np
from networks import BasicMLP
import math
from policy import Policy
from state_value_function import StateValueFunction
from torch.nn import functional as F 



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



def plot_results(training_perf):
    
    # plot the rewards
    fig, ax = plt.subplots()
    #training_perf_x = [i for i in range(len(training_perf))]
    #evaluate_perf_x = [i for i in range(len(evaluate_perf))]
    ax.plot(training_perf,'b',label='training rewards')
    #ax.plot(evaluate_perf,'r',label='evaluation rewards')
    ax.legend()
    plt.savefig('REINFORCE_baseline_rewards_over_training.png')
    
    

storage_batch = namedtuple('storage_batch',['obs','action','reward','log_prob_action'])


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
    #step_size = params.step_size # is learning rate 
    # ***** DO THE STEP_SIZE FOR THE STATE_VALUE FUNCTION AND POLICY NEED TO BE DIFFERENT ???



    policy_config = dict(network_name=params.network_name,obs_space=env.observation_space.shape,action_space=env.action_space.n)
    policy = Policy(policy_config)

    state_value_function_config = dict(network_name=params.network_name,obs_space=env.observation_space.shape,action_space=1,use_softmax=False)
    state_val_func = StateValueFunction(state_value_function_config)

    if params.optimizer == 'Adam':
        policy_optimizer = torch.optim.Adam(policy.network.parameters(),lr=params.policy_step_size)
        state_value_optimizer = torch.optim.Adam(state_val_func.network.parameters(),lr=params.state_value_step_size)
    elif params.optimizer == 'SGD':
        policy_optimizer = torch.optim.SGD(policy.network.parameters(),lr=params.policy_step_size)
        state_value_optimizer = torch.optim.SGD(state_val_func.parameters(),lr=-params.state_value_step_size)

    
    train_rewards_perf = []
    for i in range(num_episodes):
        obs = env.reset()
        log_prob_action_tracker = []
        rewards_tracker = []
        train_rewards_episode = 0
        state_values_tracker = []
        for ts in range(num_timesteps):
            action, log_prob_action = policy.select_action(obs)
            state_val = state_val_func.forward(obs)
            state_values_tracker.append(state_val)
            log_prob_action_tracker.append(log_prob_action)
            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            
            rewards_tracker.append(reward)
            train_rewards_episode += reward
            
            if done: 
                break
        train_rewards_perf.append(train_rewards_episode)
        
        update_policy(policy, state_val_func, policy_optimizer, 
            state_value_optimizer, log_prob_action_tracker,rewards_tracker, state_values_tracker, gamma)

        print(f'episode={i}  episode_reward={train_rewards_episode}')

        
    return train_rewards_perf


def update_policy(policy, state_value_func, policy_optimizer, 
    state_value_optimizer,log_prob_action_tracker,rewards_tracker, state_val_tracker, gamma):

    # compute discounted return for each timestep 
    discounted_returns = []
    for t, reward in enumerate(rewards_tracker):
        Gt = 0 
        power = 0
        for reward in rewards_tracker[t:]:
            Gt += reward * gamma ** power
            power += 1 
        discounted_returns.append(Gt)

    discounted_returns = torch.tensor(discounted_returns)

    # compute temporal difference loss 

    state_values = torch.stack(state_val_tracker)
    
    with torch.no_grad():
        temporal_diff_loss = (discounted_returns - state_values)


    # compute policy loss 

    log_prob_action_vals = torch.stack(log_prob_action_tracker)
    policy_loss = -1 * temporal_diff_loss * log_prob_action_vals
    total_policy_loss = policy_loss.sum()

    # compute state val loss 
    #state_val_loss = -1 * temporal_diff_loss * state_values 
    #total_state_val_loss = state_val_loss.sum()

    state_val_loss = F.mse_loss(discounted_returns,state_values.squeeze())
    #state_val_loss = (discounted_returns - state_values.squeeze()).sum()
    #total_state_val_loss =  (temporal_diff_loss * state_val_loss).sum()





    # gradient ascent policy
    policy_optimizer.zero_grad()
    total_policy_loss.backward()
    policy_optimizer.step()

    # gradient ascent state-value function

    state_value_optimizer.zero_grad()
    state_val_loss.backward()
    state_value_optimizer.step()


    # compute policy loss 
    #policy_loss = -1 * log_prob_action_tracker * discounted_returns
    # policy_loss = []
    # for index, log_prob_action in enumerate(log_prob_action_tracker):
    #     policy_loss.append(-1 * log_prob_action * discounted_returns[index])

    # policy_loss = torch.stack(policy_loss)
    # cumulative_policy_loss = policy_loss.sum()
    
    # optimizer.zero_grad()
    # cumulative_policy_loss.backward()
    # optimizer.step()
 
    #return 



def main():
    params = get_args()
    train_rewards_perf = train(params)
    plot_results(train_rewards_perf)




if __name__ == '__main__':
    main()