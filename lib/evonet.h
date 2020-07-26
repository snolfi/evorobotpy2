/*
 * This file belong to https://github.com/snolfi/evorobotpy
 * Author: Stefano Nolfi, stefano.nolfi@istc.cnr.it

 * evonet.h, include an implementation of a neural network policy

 * This file is part of the python module net.so that include the following files:
 * evonet.cpp, evonet.h, utilities.cpp, utilities.h, net.pxd, net.pyx and setupevonet.py
 * And can be compile with cython with the commands: cd ./evorobotpy/lib; python3 setupevonet.py build_ext –inplace; cp net*.so ../bin
 */

#ifndef EVONET_H
#define EVONET_H

class Evonet
{

public:
  // Void constructor
  Evonet();
  // Other constructor
  Evonet(int nnetworks, int heterogeneous, int ninputs, int nhiddens, int noutputs, int nlayers=1, int nhiddens2=0, int bias=0, int netType=0, int actFunct=1, int linOut=0, int wInit=0, int clip=0, int normalize=0, int randAct=0, double randActR=0.01, double wrange=1.0, int nbins=1, double low=1.0, double high=1.0);
  // Destructor
  ~Evonet();
  // Init network from file
  void initNet(char* filename);
  // set the seed
  void seed(int s);
  // Reset network
  void resetNet();
  // Copy genotype pointer
  void copyGenotype(double* genotype);
  // Copy input pointer
  void copyInput(float* input);
  // Copy output pointer
  void copyOutput(float* output);
  // Copy neuro activation pointer
  void copyNeuronact(double* na);
  // Copy normalization vector pointer
  void copyNormalization(double* no);
  // get the normalization vectors
  void getNormalizationVectors();
  // Activate network
  void updateNet();
  // Init network architecture
  void initNetArchitecture();
  // Get the number of parameters
  int computeParameters();
  // Initialize weights
  void initWeights();

  // Number of networs
  int m_nnetworks;
  // whether multiple networks are heterogeneous of not
  int m_heterogeneous;
  // Number of inputs
  int m_ninputs;
  // Number of hiddens
  int m_nhiddens;
  // Number of hiddens of the second layer
  int m_nhiddens2;
  // Number of outputs
  int m_noutputs;
  // If we use bins we multiple the number of output neurons and we store the number of 'real' output in this variable
  int m_noutputsbins;
  // Number of neurons
  int m_nneurons;
  // Number of layers
  int m_nlayers;
  // Number of free parameters
  int m_nparams;
  // Bias
  int m_bias;
  // weight range (used in uniform initialization only)
  double m_wrange;
  // whether we use bins to encode output state
  int m_nbins;
  // bins lower state
  double m_low;
  // bins higher state
  double m_high;
  // Activation of the neurons
  double* m_act;
  // Net-input of the neurons
  double* m_netinput;
  // state of the neurons (used by LSTM)
  double* m_nstate;
  // previous state of the neurons (used by LSTM)
  double* m_pnstate;
  // Network type: 0 = 'ff'; 1 = 'rec'; 2 = 'fully-rec'; 3 = 'LSTM-rec'
  int m_netType; 
  // Activation function type: 1 = 'logistic'; 2 = 'tanh'; 3 when <linOut> flag is set to 1
  int m_actFunct;
  // Activation function for the output neurons 
  int m_outType;
  // Weight initializer: 0 = 'xavier'; 1 = 'normc'
  int m_wInit;
  // Clip values flag
  int m_clip; 
  // Normalize input flag 
  int m_normalize; 
  // Random actions
  int m_randAct;
  // Random actions range: of gaussian noise
  double m_randActR;
  // Network architecture (block structure)
  int* m_netblock;
  // Number of blocks
  int m_nblocks;
  // Type of neurons: 0 = input; 1 = logistic; 2 = tanh; 3 = linOut
  int* m_neurontype;
  // normalization phase: 0=do nothing, 1=store data
  int m_normphase;
  // set the normaliztaion phase
  void normphase(int phase);
  // update normalization vectors
  void updateNormalizationVectors();
  // retrive normalization vectors from free parameters
  void setNormalizationVectors();
  // reset normalization vectors
  void resetNormalizationVectors();

private:
  // Print network architecture 
  void printNetArchitecture();
  // normalization mean
  double* m_mean;
  // normalization stdv
  double* m_std;
  // normalization sum
  double* m_sum;
  // normalization squared sum
  double* m_sumsq;
  // normalization data number
  double m_count;
  // collect normalization data
  void collectNormalizationData();
 
};

#endif
