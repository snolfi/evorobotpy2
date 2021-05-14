"""
   This file belong to https://github.com/snolfi/evorobotpy2

   It requires the pytorch and matplotlib python packages

   the script illustrates how to use pytorch to create a multi-layer neural network and how to train it
   through back-propagation to perform a regression task
   The training set included in the example permits to train a network on the XOR problem
   i.e. to react to the observations patterns 0.0 1.0 and 1.0 0.0 by producing as output 1.0
   and to react to the observation pattens 0.0 0.0 and 1.0 1.0 by producing as output 0.0
   To train the network on any other regression problem, you should only change the content of the
   training set, i.e. the input and target matrices
   More specifically, the script illustrates the usage of the torch.nn and torch.optim functions
   to create the network, to compute the loss, to compute the gradient and to update the parameters.
   Moreover, it illustrates how to access and print the parameters of the network.
   The script uses matplotlib to show the features extracted by the trained network in case of problems,
   such as the XOR, in which the observation space is two-dimensional
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy
import matplotlib.pyplot as plt

# hyper-parameters
TrainingEpochs = 5000                # the duration of the training process in epochs
nhiddens = 2                         # the number of internal neurons
DisplayEvery = 100                   # the frequency with which the program display data
                                     # the training set can be changed by modifying the inputs and targets matrices below

# define the input patters. They consists of a list of tensors. Tensors are special data type
# analogous to vectors and matrices used by pytorch which can be processed also on in parallel on GPU
inputs = torch.Tensor([[0,0],[0,1], [1,0], [1,1]])

# define and print the desired output patters. They consists of a list of tensors
targets = torch.Tensor([[0],[1],[1],[0]])

# define the class which implements the neural network
class Net(nn.Module):                         # the network class

    def __init__(self):                       # initialize the network
        super(Net, self).__init__()
        self.c1 = nn.Linear(inputs.size(1), nhiddens, True)  # creates the first layer of connection weights from 2 input to 3 internal neurons  
        self.c2 = nn.Linear(nhiddens, targets.size(1), True) # creates the second layer of connection weights from 3 internal to 1 output neurons

    def forward(self, inp):                   # update the activation of the neurons
        hid = torch.sigmoid(self.c1(inp))     # computes the activation of the internal layer by using the sigmoid activation function
        out = self.c2(hid)                    # computes the activation of the output layer by using a linear activation function
        if ((epoch % DisplayEvery) == 0):
            for i in inp:
                print("%.2f " % (i.numpy()), end="")          # print the activation of the input neurons
            print("-> ", end="")                  
            for h in hid:
                print("%.2f " % (h.detach().numpy()), end="") # print the activation of the hidden neurons
            print("-> ", end="")                  
            for o in out:
                print("%.2f " % (o.detach().numpy()), end="") # print the activation of the output neurons        
        return out

net = Net()                                      # creates the network
print("3-layers feed-forward neural network with %d input, %d internal, and %d output neurons created " % (inputs.size(1), nhiddens, targets.size(1))) 

lossfunction = nn.MSELoss()                      # creates a Mean Square Error loss function 
print("We use the Mean Squared Error Loss Function") # for other type of loss function see the pytorch documentation

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9) # create a Vanilla Gradient Descent Optimizer
print("We use the SGD Optimizer with learning rate set to 0.01 and momentum set to 0.9")
print("")

print("We train the network")
for epoch in range(0, TrainingEpochs):
    if ((epoch % DisplayEvery) == 0):
        print("Epoch %d, we print the activation of the neurons, the loss, the weights and the biases" % (epoch))
    epochloss = 0
    for inputp, targetp in zip(inputs, targets):
        optimizer.zero_grad()                  # initialize the gradient buffer to 0.0
        output = net(inputp)                   # activate the network and obtain the output
        loss = lossfunction(output, targetp)   # compute the loss
        epochloss += loss.data.numpy()         # compute the loss on the entire training set
        if ((epoch % DisplayEvery) == 0):
            print(" loss %.2f" % (loss.data.numpy()))
        loss.backward()                        # compute the gradient
        optimizer.step()                       # call the optimizer which updates the parameters by using the gradient to reduce the loss
    if ((epoch % DisplayEvery) == 0):          # display the parameters, i.e. the weights of the two layers and the biases of the internal and output layer
        w1 = net.c1.weight.detach().numpy()  
        w2 = net.c2.weight.detach().numpy()
        b1 = net.c1.bias.detach().numpy()
        b2 = net.c2.bias.detach().numpy()
        print(numpy.concatenate((w1.flatten(), w2.flatten(),b1.flatten(), b2.flatten())))
    if (epochloss < 0.0001):                   # terminate the training when the loss is lower than a threshold
        break

print("post-evaluation at the end of training")
for inputp, targetp in zip(inputs, targets):                       # post-evaluate the network at the end of training
    outputp = net(inputp)
    print("input: ", end="")
    for i in inputp:
        print("%.3f " % (i.detach().numpy()), end="")              # print the activation of the output neurons
    print("output: ", end="")
    for o in outputp:
        print("%.3f " % (o.detach().numpy()), end="")              # print the activation of the output neurons
    print("error: ", end="")
    for o, t in zip(outputp, targetp):
        print("%.3f " % (t.detach().numpy() - o.detach().numpy())) # print the error, i.e. the difference between the desired and actual outputs

# display the features extracted by the internal neurons which are used to produce the output
if (inputs.size(1) == 2):
    
    net_params = list(net.parameters())

    net_weights = net_params[0].data.numpy()
    net_bias = net_params[1].data.numpy()

    plt.scatter(inputs.numpy()[[0,-1], 0], inputs.numpy()[[0, -1], 1], c='red', s=50)  # plot the input patterns which should produce a 0.0 output
    plt.scatter(inputs.numpy()[[1,2], 0], inputs.numpy()[[1, 2], 1], c='green', s=50)  # plot the input patterns which should produce a 1.0 output

    x = numpy.zeros((nhiddens, 13))
    y = numpy.zeros((nhiddens, 13))
    for n in range(nhiddens):
        x[n] = numpy.arange(-0.1, 1.1, 0.1)
        y[n] = ((x[n] * net_weights[n,0]) + net_bias[n]) / (-net_weights[n,1]) # for each internal neuron plot a line that separate the inputs to which the neuron respond differently
    
        plt.plot(x[n], y[n])
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.show()

