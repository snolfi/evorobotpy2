# convert the observation array into a matrix with 1 column and ninputs rows
observation.resize(ninputs,1)
# compute the netinput of the first layer of neurons
Z1 = np.dot(W1, observation) + b1
# compute the activation of the first layer of neurons with the tanh function
A1 = np.tanh(Z1)
# compute the netinput of the second layer of neurons
Z2 = np.dot(W2, A1) + b2
# compute the activation of the second layer of neurons with the tanh function
A2 = np.tanh(Z2)
# if actions are discrete we select the action corresponding to the most activated unit
if (isinstance(env.action_space, gym.spaces.box.Box)):
    action = A2
else:
    action = np.argmax(A2)
