import jsonargparse
import torch 
import matplotlib.pyplot as plt 
import gym 
import numpy as np
from networks import BasicMLP
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
    plt.savefig('actor_critic_rewards_over_time.png')
    
    



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
        I = 1
        obs = env.reset()
        #log_prob_action_tracker = []
        #rewards_tracker = []
        train_rewards_episode = 0
        #state_values_tracker = []
        for ts in range(num_timesteps):
            action, log_prob_action = policy.select_action(obs)
            
            next_obs, reward, done, info = env.step(action)
            
            current_state_val = state_val_func.forward(obs)
            next_state_val = state_val_func.forward(next_obs)

            if done: 
                next_state_val = torch.tensor([0]) 
                # do logic
            
            #state_val_loss = F.mse_loss(reward + gamma * next_state_val,state_val_func.forward(obs))
            state_val_loss = F.mse_loss(reward + gamma * next_state_val, current_state_val)
            #state_val_loss *= I 
            
            delta = reward + gamma * next_state_val.item() - current_state_val.item()
            
            policy_loss = -1 * log_prob_action * delta * I 
            #state_val_loss = F.mse_loss(reward + gamma * state_val_func.forward(next_obs), state_val_func.forward(obs))

            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            state_value_optimizer.zero_grad()
            state_val_loss.backward()
            state_value_optimizer.step()

            obs = next_obs
            I *= gamma 
            
            #rewards_tracker.append(reward)
            train_rewards_episode += reward
            
            if done: 
                break
        train_rewards_perf.append(train_rewards_episode)
        
        #update_policy(policy, state_val_func, policy_optimizer, 
        #    state_value_optimizer, log_prob_action_tracker,rewards_tracker, state_values_tracker, gamma)

        print(f'episode={i}  episode_reward={train_rewards_episode}')

        
    return train_rewards_perf



def main():
    params = get_args()
    train_rewards_perf = train(params)
    plot_results(train_rewards_perf)




if __name__ == '__main__':
    main()