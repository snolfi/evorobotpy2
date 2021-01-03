import gym
import numpy as np
env = gym.make('CartPole-v0')

pvariance = 0.1     # variance of initial parameters
ppvariance = 0.02   # variance of perturbations
nhiddens = 5        # number of hidden neurons
# the number of inputs and outputs depends on the problem
# we assume that observations consist of vectors of continuous value
# and that actions can be vectors of continuous values or discrete actions
ninputs = env.observation_space.shape[0] 
if (isinstance(env.action_space, gym.spaces.box.Box)):
    noutputs = env.action_space.shape[0]
else:
    noutputs = env.action_space.n

# initialize the training parameters randomly by using a gaussian distribution with average 0.0 and variance 0.1
# biases (thresholds) are initialized to 0.0 
W1 = np.random.randn(nhiddens,ninputs) * pvariance      # first layer
W2 = np.random.randn(noutputs, nhiddens) * pvariance    # second layer
b1 = np.zeros(shape=(nhiddens, 1))                      # bias first layer
b2 = np.zeros(shape=(noutputs, 1))                      # bias second layer
