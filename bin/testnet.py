#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/snolfi/evorobotpy
   and has been written by Stefano Nolfi and Paolo Pagliuca, stefano.nolfi@istc.cnr.it, paolo.pagliuca@istc.cnr.it

   testnet.py includes and example of usage of the net component

"""

import numpy as np
import net
# parameters to be set on the basis of the environment specifications
ninputs = 4            # number of input neurons
noutputs = 2           # number of output neurons
# set configuration parameters
nnetworks = 1          # number of networks
heterogeneous = 0      # whether multiple networks are heterogeneous or not
nlayers = 1            # number of internal layers
nhiddens = 10          # number of hidden units
nhiddens2 = 0          # number of hiddens of the second layer, if any
bias = 1               # whether we have biases
architecture = 2       # full recurrent architecture
afunction = 2          # activation function of internal neurons is tanh
out_type = 3           # activation function of output neurons is linear
winit = 0              # weight initializaton is xavier
clip = 0               # whether the activation of output neuron is clipped in the [-5.0, 5.0] range
normalize = 0          # whether the activation of input is normalized
action_noise = 0       # we do not add noise to the state of output neurons
action_noise_range = 0 # the range of noise added to output units
wrange = 0.0           # range of the initial weights (when winit=2)
nbins = 1              # number of outputs neurons for output values
low = 0.0              # minimun value for output clipping, when clip=1
high = 0.0             # maximum value for output clipping, when clip=1
seed = 101             # random seed
# allocate the array for inputs, outputs, and for neuron activation
inp = np.arange(ninputs, dtype=np.float32)
out = np.arange(noutputs, dtype=np.float32)
nact = np.arange((ninputs + (nhiddens * nlayers) + noutputs), dtype=np.float64)
# create the network 
nn = net.PyEvonet(nnetworks, heterogeneous, ninputs, nhiddens, noutputs, nlayers, nhiddens2, bias, architecture,
                   afunction, out_type, winit, clip, normalize, action_noise, action_noise_range, wrange, nbins, low, high)
# allocate an array for network paramaters
nparams = nn.computeParameters()
params = np.arange(nparams, dtype=np.float64)
# allocate the array for input normalization
if (normalize == 1):
    normvector = np.arange(ninputs*2, dtype=np.float64)
else:
    normvector = None   
# pass the pointers of allocated vectors to the C++ library
nn.copyGenotype(params)
nn.copyInput(inp)
nn.copyOutput(out)
nn.copyNeuronact(nact)
if (normalize == 1):
    nn.copyNormalization(normvector)
# You can pass to the C++ library a vector of parameters (weights) with the command:
#nn.copyGenotype(params)  # the type of the vector should be np.float64
# set the seed of the library
nn.seed(seed)
#initialize the parameters randomly
nn.initWeights()
# Reset the activation of neurons to 0.0
nn.resetNet()
# For 10 steps set the input randomly, update the network, and print the output
for step in range(10):
    # generate the observation randomly
    inp = np.random.rand(ninputs).astype(np.float32)
    # pass the pointer to the input vector to the C++ library
    nn.copyInput(inp)
    # update the activation of neurons
    nn.updateNet()
    # print the activation of neurons
    print("step %d ,  print input, inputs-hidden-outputs vectors" % (step))
    print(inp)
    print(out)
    print(nact)
    


