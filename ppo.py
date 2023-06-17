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
torch.autograd.set_detect_anomaly(True)


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
    parser.add_argument('--num_iterations',default=10,type=int)
    parser.add_argument('--epsilon',default=0.2,type=float)

    return parser.parse_args()

def plot_results(training_perf):
    
    # plot the rewards
    fig, ax = plt.subplots()
    #training_perf_x = [i for i in range(len(training_perf))]
    #evaluate_perf_x = [i for i in range(len(evaluate_perf))]
    ax.plot(training_perf,'b',label='training rewards')
    #ax.plot(evaluate_perf,'r',label='evaluation rewards')
    ax.legend()
    plt.savefig('ppo_clip_rewards_over_time.png')



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
    policy_step_size = params.policy_step_size # is learning rate 
    state_value_step_size = params.state_value_step_size
    num_iterations = params.num_iterations  
    epsilon = params.epsilon


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
            sample = experience(obs,action,reward,next_obs,done,log_prob_action.item())
            trajectory_dataset.store(sample)
            obs = next_obs
            
            rewards_tracker.append(reward)
            train_rewards_episode += reward
            
            if done: 
                break
        train_rewards_perf.append(train_rewards_episode)
        
        
        update(policy,state_val_func,policy_optimizer,state_value_optimizer,trajectory_dataset,gamma,num_iterations,epsilon) 
        trajectory_dataset.clear()
        print(f'episode={i}  episode_reward={train_rewards_episode}')

    return train_rewards_perf




        

# FIX THE METHOD SIGNATURE 
def update(policy,state_val_func,policy_optimizer,state_val_func_optimizer,trajectory_dataset,gamma,num_iterations,epsilon):

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else: 
        device = torch.device('cpu')

    # compute rewards to go ( same way the discounted rewards are computed in REINFORCE)
    discounted_returns = []
    for i in range(len(trajectory_dataset)):
        Gt = 0
        power = 0 
        for index in range(i,len(trajectory_dataset)):
            Gt += trajectory_dataset[index].reward * gamma ** power
            power += 1 
        discounted_returns.append(Gt)

    
    discounted_returns = torch.tensor(discounted_returns).to(device)

    # update policy by updating PPO-clip objective 

    """
    Initial intuition for Steps 6 and 7 suggest that the idea will be very similar to what is done for 
    supervised learning. Keep in mind as you progress
    """
    advs = [] 
    for s_index in range(len(trajectory_dataset)):
        sample = trajectory_dataset[s_index]
        with torch.no_grad():
            adv = sample.reward + gamma * state_val_func.forward(sample.next_obs) - state_val_func.forward(sample.obs)
        advs.append(adv)
    advs = torch.stack(advs).to(device)
    #print('advs tensor type = ', advs.get_device())

    
    for iter_num in range(num_iterations):
        log_prob_actions = []
        curr_log_probs = []
        for s_index in range(len(trajectory_dataset)):
            sample = trajectory_dataset[s_index]
            curr_log_prob = policy.compute_log_prob_action(sample.obs,sample.action)
            curr_log_probs.append(curr_log_prob)
            log_prob_actions.append(torch.tensor([sample.log_prob_action]))

        # compute advantage estimates
        # compute TD error as advantage estimate
        
        curr_log_probs = torch.stack(curr_log_probs).to(device)
        log_prob_actions = torch.stack(log_prob_actions).to(device)
        ratio = torch.exp(curr_log_probs - log_prob_actions).to(device)
        
        #print('curr_log_probs tensor type = ', curr_log_probs.get_device())
        #print('log_prob_actions = ', log_prob_actions.get_device())
        #print('ratio=',ratio.get_device())
        
        # surr1 = ratio * advs
        # surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advs 
        # surr_final_value = (torch.min(surr1,surr2)).mean()

        clip_adv = (torch.clamp(ratio,1-epsilon,1+epsilon) * advs).to(device)
        #print('clip_adv=',clip_adv.get_device())
        #policy_loss = -(torch.min(ratio*adv,clip_adv)).mean().to(device)
        #part1 = 
        #part2 = clip_adv 
        
        policy_loss = -(torch.min(ratio * advs,clip_adv)).mean()

        #actor_loss = -1 * surr_final_value
        
        policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        policy_optimizer.step()

        state_vals = []
        for s_index in range(len(trajectory_dataset)):
            sample = trajectory_dataset[s_index]
            state_val = state_val_func.forward(sample.obs)
            state_vals.append(state_val)
            
        state_vals = torch.stack(state_vals).to(device)
        state_val_loss = F.mse_loss(state_vals.squeeze(),discounted_returns).to(device)
        

        state_val_func_optimizer.zero_grad()
        state_val_loss.backward()
        state_val_func_optimizer.step()

    return 


def main():
    params = get_args()
    train_rewards_perf = train(params)
    plot_results(train_rewards_perf)




if __name__ == '__main__':
    main()