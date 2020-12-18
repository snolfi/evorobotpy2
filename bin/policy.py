#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/snolfi/evorobotpy
   and has been written by Stefano Nolfi and Paolo Pagliuca, stefano.nolfi@istc.cnr.it, paolo.pagliuca@istc.cnr.it

   policy.py takes care of the creation and evaluation of the policy
   Requires the net.so library that can be obtained by compiling with cython the following files contained in the ./lib directory:
   evonet.cpp, evonet.h, utilities.cpp, utilities.h, net.pxd, net.pyx and setupevonet.py
   with the commands: cd ./evorobotpy/lib; python3 setupevonet.py build_ext –inplace; cp net*.so ../bin
   Also requires renderWorld.py to display neurons and to display the behavior of Er environments

"""
import net
import numpy as np
import configparser
import time
import sys

class Policy(object):
    def __init__(self, env, fileini, seed, test):
        # Copy environment
        self.env = env
        self.seed = seed
        self.test = test
        self.fileini = fileini
        self.nrobots = 1     # number of agents
        self.heterogeneous=0 # whether the policy of the agents is heterogeneous
        self.ntrials = 1     # evaluation trials
        self.nttrials = 1    # post-evaluation trials
        self.maxsteps = 1000 # max number of steps 
        self.nhiddens = 50   # number of hiddens
        self.nhiddens2 = 0   # number of hiddens of the second layer 
        self.nlayers = 1     # number of hidden layers 
        self.bias = 0        # whether we have biases
        self.out_type = 2    # output type (1=logistic,2=tanh,3=linear,4=binary)
        self.architecture =0 # Feed-forward, recurrent, or full-recurrent network
        self.afunction = 2   # activation function
        self.nbins = 1       # number of bins 1=no-beans
        self.winit = 0       # weight initialization: Xavier, normc, uniform
        self.action_noise = 0# whether we apply noise to actions
        self.action_noise_range = 0.01 # action noise range
        self.normalize = 0   # Do not normalize observations
        self.clip = 0        # clip observation
        self.displayneurons=0# Gym policies can display or the robot or the neurons activations
        self.wrange = 1.0    # weight range, used in uniform initialization only
        self.low = -1.0      # mimimum activation
        self.high = 1.0      # maximum activation
        
        # Read configuration file
        self.readConfig()
        # Display info
        print("Policy: episodes %d postevaluation episodes %d maxsteps %d input normalization %d nrobots %d heterogeneous %d" % (self.ntrials, self.nttrials, self.maxsteps, self.normalize, self.nrobots, self.heterogeneous))        
        self.ob = np.arange(self.ninputs * self.nrobots, dtype=np.float32)  # allocate observation vector
        self.ac = np.arange(self.noutputs * self.nrobots, dtype=np.float32) # allocate action vector
        if (self.nbins == 1):                                          # allocate neuron activation vector
            self.nact = np.arange((self.ninputs + (self.nhiddens * self.nlayers) + self.noutputs) * self.nrobots, dtype=np.float64)
        else:
            self.nact = np.arange((self.ninputs + (self.nhiddens * self.nlayers) + (self.noutputs * self.nbins)) * self.nrobots, dtype=np.float64)
        if (self.normalize == 1):                                      # allocate normalization vector
            self.normvector = np.arange(self.ninputs*2, dtype=np.float64)
        else:
            self.normvector = None
        # Initialize the neural network
        self.nn = net.PyEvonet(self.nrobots, self.heterogeneous, self.ninputs, (self.nhiddens * self.nlayers), self.noutputs, self.nlayers, self.nhiddens2, self.bias, self.architecture, self.afunction, self.out_type, self.winit, self.clip, self.normalize, self.action_noise, self.action_noise_range, self.wrange, self.nbins, self.low, self.high)
        # Initialize policy parameters (control and eventually morphological)
        self.nparams = self.nn.computeParameters()
        try:
            self.nmorphparams = self.env.getNumParams()
            self.nparams += self.nmorphparams
        except:
            self.nmorphparams = 0
        self.params = np.arange(self.nparams, dtype=np.float64)
        self.normphase = 0     # whether we collect data to update normalization in the current episode
        self.norm_prob = 0.01  # the fraction of episodes in which normalization data is collected           
        self.nn.copyGenotype(self.params)   # pass the pointer to the parameters to evonet
        self.nn.copyInput(self.ob)          # pass the pointer to the observation vector to evonet
        self.nn.copyOutput(self.ac)         # pass the pointer to the action vector to evonet
        self.nn.copyNeuronact(self.nact)    # pass the pointer to the neuron activation vector to evonet
        if (self.normalize == 1):           # pass the pointer to the nornalization vector to evonet
            self.nn.copyNormalization(self.normvector)
        self.nn.seed(self.seed)             # initilaize the seed of evonet
        self.nn.initWeights()               # initialize the control parameters of evonet 
        if self.nmorphparams > 0:           # initialize the morphological parameter to 0, if any
            for i in range(self.nmorphparams):
                self.params[(self.nparams - self.nmorphparams + i)] = 0.0
            self.env.setParams(self.params[-self.nmorphparams:])
         
    def reset(self):
        self.nn.seed(self.seed)             # set the seed of evonet
        self.nn.initWeights()               # call the evonet function that initialize the parameters
        if (self.normalize == 1):           # re-initialize the normalization vector
            self.nn.resetNormalizationVectors()

    # virtual function, implemented in sub-classes
    def rollout(self, render=False, seed=None):
        raise NotImplementedError

    def set_trainable_flat(self, x):        
        self.params = np.copy(x)
        self.nn.copyGenotype(self.params)   # copy a vector of parameters in the evonet parameter vector
        if self.nmorphparams > 0:
            self.env.setParams(self.params[-self.nmorphparams:]) # set the morphology with the morphological parameters

    def get_trainable_flat(self):
        return self.params                  # return the evonet vector of parameters


    def readConfig(self):                   # load hyperparameters from the [POLICY] section of the ini file
        
        config = configparser.ConfigParser()
        config.read(self.fileini)
        options = config.options("POLICY")
        for o in options:
          found = 0
          if o == "nrobots":
              self.nrobots = config.getint("POLICY","nrobots")
              found = 1
          if o == "heterogeneous":
              self.heterogeneous = config.getint("POLICY","heterogeneous")
              found = 1
          if (o == "episodes"):
              self.ntrials = config.getint("POLICY","episodes")
              found = 1
          if (o == "pepisodes"):
              self.nttrials = config.getint("POLICY","pepisodes")
              found = 1
          if (o == "maxsteps"):
              self.maxsteps = config.getint("POLICY","maxsteps")
              found = 1
          if (o == "nhiddens"):
              self.nhiddens = config.getint("POLICY","nhiddens")
              found = 1
          if (o == "nhiddens2"):
              self.nhiddens2 = config.getint("POLICY","nhiddens2")
              found = 1
          if (o == "nlayers"):
              self.nlayers = config.getint("POLICY","nlayers")
              found = 1
          if (o == "bias"):
              self.bias = config.getint("POLICY","bias")
              found = 1
          if (o == "out_type"):
              self.out_type = config.getint("POLICY","out_type")
              found = 1
          if (o == "nbins"):
              self.nbins = config.getint("POLICY","nbins")
              found = 1
          if (o == "afunction"):
              self.afunction = config.getint("POLICY","afunction")
              found = 1
          if (o == "architecture"):
              self.architecture = config.getint("POLICY","architecture")
              found = 1
          if (o == "winit"):
              self.winit = config.getint("POLICY","winit")
              found = 1
          if (o == "action_noise"):
              self.action_noise = config.getint("POLICY","action_noise")
              found = 1
          if (o == "action_noise_range"):
              self.action_noise_range = config.getfloat("POLICY","action_noise_range")
              found = 1
          if (o == "normalize"):
              self.normalize = config.getint("POLICY","normalize")
              found = 1
          if (o == "clip"):
              self.clip = config.getint("POLICY","clip")
              found = 1
          if (o == "wrange"):
              self.wrange = config.getint("POLICY","wrange")
              found = 1  
          if (found == 0):
              print("\033[1mOption %s in section [POLICY] of %s file is unknown\033[0m" % (o, self.fileini))
              sys.exit()

    @property
    def get_seed(self):
        return self.seed

# Bullet use float32 values for observation and action vectors (the same type used by evonet)
# create a new observation vector each step, consequently we need to pass the pointer to evonet each step 
# Use renderWorld to show the activation of neurons
class BulletPolicy(Policy):
    def __init__(self, env, filename, seed, test):
        self.ninputs = env.observation_space.shape[0]      # only works for problems with continuous observation space
        self.noutputs = env.action_space.shape[0]          # only works for problems with continuous action space
        Policy.__init__(self, env, filename, seed, test)                            
    
    def rollout(self, ntrials, render=False, seed=None):   # evaluate the policy for one or more episodes 
        rews = 0.0                      # summed reward
        steps = 0                       # steps performed
        if (self.test == 2):
            self.objs = np.arange(10, dtype=np.float64) # if the policy is used to test a trained agent and to visualize the neurons, we need initialize the graphic render  
            self.objs[0] = -1
            import renderWorld
        if seed is not None:
            self.env.seed(seed)          # set the seed of the environment that impacts on the initialization of the robot/environment
            self.nn.seed(seed)           # set the seed of evonet that impacts on the noise eventually added to the activation of the neurons
        for trial in range(ntrials):
            self.ob = self.env.reset()   # reset the environment
            self.nn.resetNet()           # reset the activation of the neurons (necessary for recurrent policies)
            rew = 0
            t = 0
            while t < self.maxsteps:
                self.nn.copyInput(self.ob)                    # copy the pointer to the observation vector to evonet
                self.nn.updateNet()                           # update the activation of the policy
                self.ob, r, done, _ = self.env.step(self.ac)  # perform a simulation step
                rew += r
                t += 1
                if (self.test > 0):
                    if (self.test == 1):
                        self.env.render(mode="human")
                        time.sleep(0.05)
                    if (self.test == 2):
                        info = 'Trial %d Step %d Fit %.2f %.2f' % (trial, t, r, rew)
                        renderWorld.update(self.objs, info, self.ob, self.ac, self.nact)
                if done:
                    break
            if (self.test > 0):
                print("Trial %d Fit %.2f Steps %d " % (trial, rew, t))
            steps += t
            rews += rew
        rews /= ntrials                # Normalize reward by the number of trials
        if (self.test > 0 and ntrials > 1):
            print("Average Fit %.2f Steps %.2f " % (rews, steps/float(ntrials)))
        return rews, steps
    
# Gym policies use float64 observation and action vectors (evonet use float32)
# create a new observation vector each step, consequently we need to pass the pointer to evonet each step 
# Use renderWorld to show the activation of neurons
class GymPolicy(Policy):
    def __init__(self, env, filename, seed, test):
        self.ninputs = env.observation_space.shape[0]      # only works for problems with continuous observation space
        self.noutputs = env.action_space.shape[0]          # only works for problems with continuous action space
        Policy.__init__(self, env, filename, seed, test)

    def rollout(self, ntrials, render=False, seed=None):   # evaluate the policy for one or more episodes 
        rews = 0.0                    # summed rewards
        steps = 0                     # step performed
        if (self.test == 2):          # if the policy is used to test a trained agent and to visualize the neurons, we need initialize the graphic render  
            import renderWorld
            self.objs = np.arange(10, dtype=np.float64)   
            self.objs[0] = -1 
        if seed is not None:
            self.env.seed(seed)          # set the seed of the environment that impacts on the initialization of the robot/environment
            self.nn.seed(seed)           # set the seed of evonet that impacts on the noise eventually added to the activation of the neurons
        for trial in range(ntrials):
            self.ob = self.env.reset()   # reset the environment at the beginning of a new episode
            self.nn.resetNet()           # reset the activation of the neurons (necessary for recurrent policies)
            rew = 0.0
            t = 0
            while t < self.maxsteps:
                self.nn.copyInput(np.float32(self.ob))        # copy the pointer to the observation vector to evonet and convert from float64 to float32
                self.nn.updateNet()                           # update the activation of the policy
                self.ob, r, done, _ = self.env.step(self.ac)  # perform a simulation step
                rew += r
                t += 1
                if (self.test > 0):
                    if (self.test == 1):
                        self.env.render()
                        time.sleep(0.05)
                    if (self.test == 2):
                        info = 'Trial %d Step %d Fit %.2f %.2f' % (trial, t, r, rew)
                        renderWorld.update(self.objs, info, self.ob, self.ac, self.nact)
                if done:
                    break
            if (self.test > 0):
                print("Trial %d Fit %.2f Steps %d " % (trial, rew, t))
            steps += t
            rews += rew
        rews /= ntrials               # Normalize reward by the number of trials
        if (self.test > 0 and ntrials > 1):
            print("Average Fit %.2f Steps %.2f " % (rews, steps/float(ntrials)))
        return rews, steps

# GymDiscr policies use float64 observation and action vectors (evonet use float32) and use discrete action vectors
# create a new observation vector each step, consequently we need to pass the pointer to evonet each step 
# Use renderWorld to show the activation of neurons
class GymPolicyDiscr(Policy):
    def __init__(self, env, filename, seed, test):
        self.ninputs = env.observation_space.shape[0]       # only works for problems with continuous observation space
        self.noutputs = env.action_space.n                  # only works for problems with discrete action space
        Policy.__init__(self, env, filename, seed, test)
        
    def rollout(self, ntrials, render=False, seed=None):
        rews = 0.0                # summed rewards
        steps = 0                 # step performed
        if (self.test == 2):      # if the policy is used to test a trained agent and to visualize the neurons, we need initialize the graphic render  
            import renderWorld
            self.objs = np.arange(10, dtype=np.float64)   
            self.objs[0] = -1 
        if seed is not None:
            self.env.seed(seed)   # set the seed of the environment that impacts on the initialization of the robot/environment
            self.nn.seed(seed)    # set the seed of evonet that impacts on the noise eventually added to the activation of the neurons
        for trial in range(ntrials):
            self.ob = self.env.reset()                   # reset the environment at the beginning of a new episode
            self.nn.resetNet()                           # reset the activation of the neurons (necessary for recurrent policies)
            rew = 0.0
            t = 0
            while t < self.maxsteps:
                self.nn.copyInput(np.float32(self.ob))      # copy the pointer to the observation vector to evonet and convert from float64 to float32
                self.nn.updateNet()                         # update the activation of the policy
                action = np.argmax(self.ac)                 # select the action that corresponds to the most activated output neuron
                self.ob, r, done, _ = self.env.step(action) # perform a simulation step
                rew += r
                t += 1
                if render:
                    if (self.test == 1):
                        self.env.render()
                        time.sleep(0.05)
                    if (self.test == 2):
                        info = 'Trial %d Step %d Fit %.2f %.2f' % (trial, t, r, rew)
                        renderWorld.update(self.objs, info, self.ob, self.ac, self.nact)
                if done:
                    break
            if (self.test > 0):
                print("Trial %d Fit %.2f Steps %d " % (trial, rew, t))
            steps += t
            rews += rew
        rews /= ntrials          # Normalize reward by the number of trials
        if (self.test > 0 and ntrials > 1):
            print("Average Fit %.2f Steps %d " % (rews, steps/float(ntrials)))
        return rews, steps


# ER policies use float32 observation and action vectors (evonet also use float32) 
# use the same observation, action and done vectors (no need to re-pass the pointer)
# Use renderWorld to show the activation of neurons
class ErPolicy(Policy):
    def __init__(self, env, filename, seed, test):
        self.ninputs = env.ninputs               # only works for problems with continuous observation space
        self.noutputs = env.noutputs             # only works for problems with continuous observation space
        Policy.__init__(self, env, filename, seed, test)
        self.done = np.arange(1, dtype=np.int32) # allocate a done vector
        env.copyObs(self.ob)                     # pass the pointer to the observation vector to the Er environment
        env.copyAct(self.ac)                     # pass the pointer to the action vector to the Er environment
        env.copyDone(self.done)                  # pass the pointer to the done vector to the Er environment    
    # === Rollouts/training ===
    def rollout(self, ntrials, render=False, seed=None):
        rews = 0.0                               # summed reward
        steps = 0                                # steps performed
        if seed is not None:
            self.env.seed(seed)                  # set the seed of the environment that impacts on the initialization of the robot/environment
            self.nn.seed(seed)                   # set the seed of evonet that impacts on the noise eventually added to the activation of the neurons
        if (self.test > 0):                      # if the policy is used to test a trained agent and to visualize the neurons, we need initialize the graphic render  
            self.objs = np.arange(1000, dtype=np.float64) 
            self.objs[0] = -1
            self.env.copyDobj(self.objs)
            import renderWorld
        for trial in range(ntrials):
            self.env.reset()                     # reset the environment at the beginning of a new episode
            self.nn.resetNet()                   # reset the activation of the neurons (necessary for recurrent policies)
            rew = 0.0
            t = 0
            while t < self.maxsteps:
                self.nn.updateNet()              # update the activation of the policy
                rew += self.env.step()           # perform a simulation step
                t += 1
                if (self.test > 0):
                    self.env.render()
                    info = 'Trial %d Step %d Fit %.2f %.2f' % (trial, t, rew, rews)
                    renderWorld.update(self.objs, info, self.ob, self.ac, self.nact)
                if self.done:
                    break
            if (self.test > 0):
                print("Trial %d Fit %.2f Steps %d " % (trial, rew, t))
            steps += t
            rews += rew
        rews /= ntrials                         # Normalize reward by the number of trials
        if (self.test > 0 and ntrials > 1):
            print("Average Fit %.2f Steps %.2f " % (rews, steps/float(ntrials)))
        return rews, steps
        

