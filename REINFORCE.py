import jsonargparse
import torch 
import matplotlib.pyplot as plt 
import gym 
from collections import namedtuple
import numpy as np
from networks import BasicMLP
import math
from policy import Policy


def get_args():
   
    parser = jsonargparse.ArgumentParser()
    parser.add_argument('--config',action=jsonargparse.ActionConfigFile)
    parser.add_argument('--network_name',default='BasicMLP',type=str)
    parser.add_argument('--env_name',default='CartPole-v1',type=str)
    parser.add_argument('--num_episodes',default=1000,type=int)
    parser.add_argument('--num_timesteps',default=10000,type=int)
    parser.add_argument('--gamma',type=float)
    parser.add_argument('--step_size',type=float)
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
    plt.savefig('rewards_over_training_eval.png')
    
    

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
    step_size = params.step_size # is learning rate 


    policy_config = dict(network_name=params.network_name,obs_space=env.observation_space.shape,action_space=env.action_space.n)
    policy = Policy(policy_config)

    if params.optimizer == 'Adam':
        optimizer = torch.optim.Adam(policy.network.parameters(),lr=step_size)
    elif params.optimizer == 'SGD':
        optimizer = torch.optim.SGD(policy.network.parameters(),lr=step_size)

    
    train_rewards_perf = []
    for i in range(num_episodes):
        obs = env.reset()
        log_prob_action_tracker = []
        rewards_tracker = []
        train_rewards_episode = 0

        for ts in range(num_timesteps):
            action, log_prob_action = policy.select_action(obs)
            log_prob_action_tracker.append(log_prob_action)
            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            
            rewards_tracker.append(reward)
            train_rewards_episode += reward
            
            if done: 
                break
        train_rewards_perf.append(train_rewards_episode)
        
        update_policy(policy, optimizer, log_prob_action_tracker,rewards_tracker,gamma)

        print(f'episode={i}  episode_reward={train_rewards_episode}')

        
    return train_rewards_perf


def update_policy(policy,optimizer,log_prob_action_tracker,rewards_tracker,gamma):

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


    # compute policy loss 
    #policy_loss = -1 * log_prob_action_tracker * discounted_returns
    policy_loss = []
    for index, log_prob_action in enumerate(log_prob_action_tracker):
        policy_loss.append(-1 * log_prob_action * discounted_returns[index]) # negative one -- for stochastic gradient ascent not descent 

    policy_loss = torch.stack(policy_loss)
    cumulative_policy_loss = policy_loss.sum()
    
    optimizer.zero_grad()
    cumulative_policy_loss.backward()
    optimizer.step()
 
    return 



def main():
    params = get_args()
    train_rewards_perf = train(params)
    plot_results(train_rewards_perf)




if __name__ == '__main__':
    main()