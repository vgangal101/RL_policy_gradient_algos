import jsonargparse
import torch 
import matplotlib.pyplot as plt 
import gym 

def get_args():
    parser = jsonargparse.ArgumentParser()
    parser.add_argument()


    return parser.parse_args()


def plot_results():
    return 



def train_REINFORCE():
    """
    Train ONLY on cartpole or classic control tasks 
    """

    # initialize environment. using args 


    num_episodes = None 
    num_timesteps = None 
    gamma = None # make it a default of 0.99 
    step_size = None 

    
    for i in range(num_episodes):
        obs = env.reset()
        storage = []
        while not done:
            action = policy(obs)
            next_obs, done, reward, info = env.step(action)
            storage.append((obs,action,reward))
            obs = next_obs 
        for batch in storage:
            # compute RETURN 
            # compute weight update. 
            pass 
              





    return 

def evaluate():
    return 

def main():
    return 




if __name__ == '__main__':
    main()