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

    return parser.parse_args()



def plot_results(training_perf, evaluate_perf):
    
    # plot the rewards
    fig, ax = plt.subplots()
    #training_perf_x = [i for i in range(len(training_perf))]
    #evaluate_perf_x = [i for i in range(len(evaluate_perf))]
    ax.plot(training_perf,'b',label='training rewards')
    ax.plot(evaluate_perf,'r',label='evaluation rewards')
    ax.legend()
    plt.savefig('rewards_over_training_eval.png')
    
    

storage_batch = namedtuple('storage_batch',['obs','action','reward'])


def train(params):
    """
    Train ONLY on cartpole or classic control tasks 
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

    optimizer = torch.optim.SGD(policy.network.parameters(),lr=step_size)

    eval_reward_perf = []
    train_rewards_perf = []
    for i in range(num_episodes):
        obs = env.reset()
        storage = []
        train_rewards_episode = 0
        for ts in range(num_timesteps):
            action = policy.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            train_rewards_episode += reward
            storage.append(storage_batch(obs,action,reward))
            obs = next_obs 
            if done: 
                break
        train_rewards_perf.append(train_rewards_episode)
        
        for t, batch in enumerate(storage):
            # compute RETURN
            G = 0 
            for k in range(t,len(storage)):
                G += math.pow(gamma,k-t-1)  * storage[k].reward

            # compute log prob of actions given state , neural net
            log_state_action_value = policy.compute_log_prob_action(batch.obs,batch.action)
            policy_loss = -1 * log_state_action_value * G 
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

        if i % 5 == 0: 
            reward_mean_performance = evaluate(env,policy,5)
            print(f'episode {i} reward_perf={reward_mean_performance}')
            eval_reward_perf.append(reward_mean_performance)
            
    torch.save(policy.network,'REINFORCE_model_weights.pth')
    return train_rewards_episode, eval_reward_perf

def evaluate(env,policy,num_episodes):
    rewards_perf = []
    for i in range(num_episodes):
        obs = env.reset()
        done = False 
        episode_reward = 0 
        while not done:
            with torch.no_grad():
                action = policy.select_action(obs)
            next_obs, done, reward, info = env.step(action)
            episode_reward += reward 
            obs = next_obs  
        rewards_perf.append(episode_reward)
    
    return sum(rewards_perf) / len(rewards_perf)

def main():
    params = get_args()
    train_rewards_perf, eval_rewards_perf = train(params)
    plot_results(train_rewards_perf,eval_rewards_perf)




if __name__ == '__main__':
    main()