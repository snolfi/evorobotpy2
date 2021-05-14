#!/usr/bin/env python3

"""
   This file belong to https://github.com/snolfi/evorobotpy2
   It contain the example of a crossentropy algorithm applied to the Cart-Pole-v0 environment adapted from an example included
   in Lapan, M. (2018). Deep Reinforcement Learning Hands-On: Apply modern RL methods, with deep Q-networks, value iteration, policy gradients, TRPO, AlphaGo Zero and more. Packt Publishing Ltd.

   It requires the pytorch python package

   The training is realized by performing N evaluation episodes in each training epoch
   and by using the observations and the associated selected actions of the best episodes to train the policy network (the data of the other episodes is discarded)
   The variations which enable the discovery of better solutions are introduced by selecting the actions randomly with the probabities determined by the policy network
"""

import gym
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# hyper-parameters
NINTERNAL = 128              # Number of internal neurons
NEPISODES = 16               # Number of episode performed in each iteration
PERCENTILE = 70              # Fraction of the best evaluation episode used to train the policy network


class Net(nn.Module):        # The policy network class
    def __init__(self, obs_size, ninternal, nactions):  
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, ninternal),
            nn.ReLU(),                        # The internal neurons use the ReLU activation function
            nn.Linear(ninternal, nactions)    # The output neurons encode the probability if execution of the n different possible actions
        )

    def forward(self, x):
        return self.net(x)

# define two tuple helper classes by using the collection package
Episode = namedtuple('Episode', field_names=['reward', 'steps'])               # store the total undiscounted reward and the steps, i.e. the list of observations and actions of an episode
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action']) # list of observations and action of an episode


def rollout(env, net, nepisodes): # carry on the evaluation episodes and store data (cumulative reward, observations and actions)
    rolldata = []           # initialize the rollout data list
    episode_reward = 0.0    # initialize the episode_reward
    episode_steps = []      # initialize the list containing the rollout data
    obs = env.reset()       # initialize the episode
    sm = nn.Softmax(dim=1)  # create a softmax class
    while True:
        obs_v = torch.FloatTensor([obs])                                    # convert the observation vector to a tensor vector
        act_probs_v = sm(net(obs_v))                                        # compute the action vector with the policy network and convert the action vector to a probability distribution with softmax 
        act_probs = act_probs_v.data.numpy()[0]                             # convert the action tensor vector in a numpy vector
        action = np.random.choice(len(act_probs), p=act_probs)              # choses the action randomly by using the action probabilities
        next_obs, reward, is_done, _ = env.step(action)                     # perform a step
        episode_reward += reward                                            # update the total undiscounted reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))   # append the observation and the action to the data
        if is_done:                                                         # at the end of the episode:
            rolldata.append(Episode(reward=episode_reward, steps=episode_steps))   # create the data of the episode, i.e. the summed undiscounted reward and the list of observation and actions
            episode_reward = 0.0                                                # re-initialize the reward
            episode_steps = []                                                  # re-initialize the steps
            next_obs = env.reset()                                              # re-initialize the episode
            if len(rolldata) == nepisodes:
                yield rolldata                                                  # return a list of data collected during the rollout 
                rolldata = []
        obs = next_obs                                                       # set the next observation


def filter_rollout(rolldata, percentile): # filter out the data of the worse episode
    rewards = list(map(lambda s: s.reward, rolldata))     # extracts the list of comulative rewards collected in the corresponding episodes
    reward_bound = np.percentile(rewards, percentile)  # computes the minimum reward which episodes should have to be used for training the policy network
    reward_mean = float(np.mean(rewards))              # computes the reward mean

    train_obs = []                                     # initializes the matrix of observation to be used for training
    train_act = []                                     # initializes the matrix of actions to be used for training
    for episode in rolldata:
        if episode.reward < reward_bound:              # check whether the reward of the episode exceed the minimal bound
            continue                                                        # data of low performing episodes are discarded
        train_obs.extend(map(lambda step: step.observation, episode.steps)) # store the observation for training
        train_act.extend(map(lambda step: step.action, episode.steps))      # store the actions for training

    train_obs_v = torch.FloatTensor(train_obs)         # transform the observation matrix in a tensor
    train_act_v = torch.LongTensor(train_act)          # transform the action matrix in a tensor
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    print("CartPole-v0 trained through a crossentropy reinforcement learning algorithm")
    print("Nepisodes x epoch: %d fraction episode discarded %.2f" % (NEPISODES, PERCENTILE))
    env = gym.make("CartPole-v0")               # create the environment
    nsensory = env.observation_space.shape[0]   # extract the number of sensory neurons
    nmotor = env.action_space.n                 # extract the number of motor neurons by assuming that the action space of the environment is discrete

    net = Net(nsensory, NINTERNAL, nmotor)      # create the network policy
    print("The policy network has %d sensory, %d internal and %d motor neurons" % (nsensory, NINTERNAL, nmotor))
    objective = nn.CrossEntropyLoss()           # the Cross Entropy Loss function combine cross entropy and softmax. It works with raw values (digits) thus avoiding the need to add a softmax layer in the policy network 
    optimizer = optim.Adam(params=net.parameters(), lr=0.01) # initialize the Adap stochastic optimizer
    print("")

    for epoch, rolldata in enumerate(rollout(env, net, NEPISODES)):
        obs_v, acts_v, reward_b, reward_m = filter_rollout(rolldata, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("epoch %d: loss=%.3f, reward_mean=%.1f, reward_threshould=%.1f" % (epoch, loss_v.item(), reward_m, reward_b))
        if reward_m > 199:
            print("Solved!")
            break
