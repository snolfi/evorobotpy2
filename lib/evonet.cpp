/*
 * This file belong to https://github.com/snolfi/evorobotpy
 * Author: Stefano Nolfi, stefano.nolfi@istc.cnr.it

 * evonet.cpp, include an implementation of a neural network policy

 * This file is part of the python module net.so that include the following files:
 * evonet.cpp, evonet.h, utilities.cpp, utilities.h, net.pxd, net.pyx and setupevonet.py
 * And can be compile with cython with the commands: cd ./evorobotpy/lib; python3 setupevonet.py build_ext –inplace; cp net*.so ../bin
 */


/*
 * evonet.cpp and evonet.h implement a tool for creating and updating a neural network or a series of neural network with the same architecture
 *
 * the library include the following method:
 * Evonet::Evonet()                    constructs the network
 * Evonet::resetNet()                  reset the activation of the neurons at the beginning of an episode
 * Evonet::updateNet()                 update the activation of the network by using as input the observation and by copy the state of the motor neurons in the action vector
 * Evonet::computeParameters()         calculate the number of parameters required by a constructed network
 * Evonet::initWeights()               initialize the parameters randomly
 * Evonet::copyGenotype()              pass the pointer to the vector that contains the free parameters (the vectors should be allocated elsewhere)
 * Evonet::copyInput()                 pass the pointer to the vector that contained the observation (the vectors should be allocated elsewhere)
 * Evonet::copyOutput()                pass the pointer to the vector that contained the action (the vectors should be allocated elsewhere)
 * Evonet::copyNeuronact()             pass the pointer to the vector that contained the activation of the neurons (the vectors should be allocated elsewhere)
 * Evolnet::copyNormalization()        pass the pointer to the vector that is used to normalize the observations 
 * Evonet::resetNormalizationVectors() reset the normnalization vector (used only when observations are normalized)
 * Evonet::normphase()                 turn on or off the collection of data for normalization
 *
 * the Evonet::Evonet() constructor receives the following parameters:
 * nnetworks        the number of neural network to be created
 * heterogeneous    whether the multiple networks are heterogenous (i.e. have different connection weights or not)
 * ninputs          the number of sensory neurons or inputs
 * nhiddens         the number of internal neurons or hiddens
 * noutputs         the number of motor neurons or outputs
 * nlayers          the number of layers
 * nhiddens2        the number of hiddens of the second layer
 * bias             whether internal and motor type have neurons
 * netType          the architecture (0=feedforward 1=recurrent 2=fullrecurrent 3=lstm recurrent
 * actFunct         the activation function (1=logistic 2=tanh 3=linear)
 * outType          the activation function of the motor neurons (1=logistic 2=tanh 3=linear)
 * wInit            the weights initialization method 0=xavier 1=norm incoming 2=uniform
 * clip             whether observation are clipped in the range [-5.0, 5.0]
 * normalize        whether the oservations are normalized (see [1])
 * randAct          the type of noise applied to motor neurons 0-nonoise 1=gaussian 2=gaussian-parametric (as in diagonal gaussian policy)
 * randActR         the range of uniform noise [used only when randact=1]
 * wrange           the range of weights initailization [used only when wInit=2]
 * nbins            if difefrent from 1, number of bins used to encode motors (see [1])
 * low              minimum values for bins
 * high             maximum value for bins
 *
 * the Evonet::Evonet() constructor create a series of blocks (tuples of 3-5 integers) that describe the neurons to be updated and the connectivity among neurons
 * the first number of each tuple indicate the type of the block (0=connection block, 1=update block, 2=special connection block for LSTM networks)
 * the second number is the index of the first neuron and the third number of number of neurons forming the block
 * in the case of connection block, the fourth number is the index of the first pre-synaptic neuron and the fifth number is the number of presynaptic neurons                                     
 * neurons are arranged in the following order, first the sensory neurons, then the internal neurons, and finally the motor neurons.
 * for example in the case of a feedword network with 10 sensory neurons, a first layer of 20 internal neurons, a second layer of 10 internal neurons, and a motor laters with 5 motor neurons
 * the constructor will create the following blocks:
 * 
 * 1  0 10        = update block that is used to update the activation of sensory neurons [0-9]
 * 0 10 20 0 10   = connection block that is used to update the netinput of the internal layer of the first neurons [10-29] that receive connections from neurons [0-9]
 * 1 10 20        = update block that is used to update the activation of the first layet of internal neurons [10-29]
 * 0 30 10 10 20  = connection block that is used to update the netinput of the second layer of internal neurons [10-29] that receive connections from neurons [0-9]
 * 1 30 10        = update block that is used to update the activation of the second layet of internal neurons [10-30]
 * 0 40 5 30 10   = connection block that is used to update the netinput of the motor neurons [40-44] that receive connections from neurons [30-39]
 * 1 40 5         = update block that is used to update the activation of the motor neurons [40-44]
 *
 * the function to be used to activate each neuron is indicated in the vector m_neurontype
 *
 * the Evonet::updateNet() function update the activation of the neurons and copy the activation of the motor neurons on the actin vector
 * the observation vector should be updated elsewhere before the function is called
 * for example in the case of the feedfoward network illustrated above it does the following:
 * update the state of the first 10 sensory neurons by setting the state of each neuron to the state of the corresponding observation value
 * compute the netinput of the successive 20 neurons, i.e. of the neurons forming the first internal layer
 * update the activation of the first layer of internal neurons on the basis of the netinput calculated above
 * compute the netinput of the successive 10 neurons, i.e. of the neurons forming the second internal layer
 * update the activation of the second layer of internal neurons on the basis of the netinput calculated above
 * compute the netinput of the successive 5 neurons, i.e. of the neurons forming the motor layer
 * update the activation of motor neurons on the basis of the netinput calculated above
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <string.h>
#include "evonet.h"
#include "utilities.h"

#define MAX_BLOCKS 20
#define MAXN 10000
#define CLIP_VALUE 5.0

// set to 1 to print additional information about the architecture
int verbose = 1;

// Local random number generator
RandomGenerator* netRng;

// pointers to vectors wrapped in python
double *cgenotype = NULL;       // parameters of the policy
float *cobservation = NULL;     // observation
float *caction = NULL;          // action vector
double *neuronact = NULL;       // actiovation (output) of the neurons
double *cnormalization = NULL;  // vector used to normalize the observation

// logistic activation functiom
double logistic(double f)
{
  return ((double) (1.0 / (1.0 + exp(-f))));
}

// hyperbolic tangent activation function
double tanh(double f)
{

  if (f > 10.0)
    return 1.0;
  else if (f < - 10.0)
    return -1.0;
  else
    return ((double) ((1.0 - exp(-2.0 * f)) / (1.0 + exp(-2.0 * f))));
 
}

// linear activation function
double linear(double f)
{
  return (f);
}


// constructor of the policy
Evonet::Evonet()
{
  m_ninputs = 0;   // number of input neurons
  m_nhiddens = 0;  // number of internal neurons
  m_nhiddens2 = 0; // number of hiddens of the second layer
  m_noutputs = 0;  // number of motor neurons
  m_nneurons = (m_ninputs + m_nhiddens + m_noutputs);  // total number of neurons
  m_nlayers = 1;   // number of layers
  m_bias = 0;      // whether internal and motor neurons have biases
  m_netType = 0;   // type of policy network
  m_actFunct = 2;  // activation function, 1=logistic 2=tanh 3=linear (default 2)
  m_outType = 0;   // activation function for the motor neurons, 1=logistic, 2=tanh, 3=linear, 4=binary (default 2)
  m_wInit = 0;     // weights initialization mode, 0=xavier 1=norm incoming 2=uniform (default 0)
  m_clip = 0;      // whether we clip observation in [-5,5] (default 0)
  m_normalize = 0; // whether or not the input observations are normalized (default 1)
  m_randAct = 0;   // action noise 0=none, 1=gaussian 2=gaussian-parametric (default 0)
  m_wrange = 1.0;  // action noise range (default 0.01)
  m_nbins = 1;     // whether actions are encoded in bins
  m_low = -1.0;    // minimum value for observations and actions
  m_high = 1.0;    // maximum value for observations and actions
 
    
  netRng = new RandomGenerator(time(NULL));  // create a random generator
  m_netinput = new double[MAXN];             // allocate the vector of neurons netinputs 
  m_netblock = new int[MAX_BLOCKS * 5];      // allocate the vector containing the description of the architecture
  m_nblocks = 0;                             // number of blocks describing the architecture
  m_neurontype = new int[MAXN];              // allocate the vector that encode the type of each neuron

}

// set the seed
void Evonet::seed(int s)
{
  netRng->setSeed(s);
}


// initialize the network
Evonet::Evonet(int nnetworks, int heterogeneous, int ninputs, int nhiddens, int noutputs, int nlayers, int nhiddens2, int bias, int netType, int actFunct, int outType, int wInit, int clip, int normalize, int randAct, double randActR, double wrange, int nbins, double low, double high)
{
    
  m_nnetworks = nnetworks;           // number of networks
  m_heterogeneous = heterogeneous;   // whether the networks are heterogeneous or homogeneous
  m_nbins = nbins;                   // number of bins, 1 is equivalent to no bins
  if (m_nbins < 1 || m_nbins > 20) 
        m_nbins = 1;
  m_ninputs = ninputs;
  m_nhiddens = nhiddens;
  m_nhiddens2 = nhiddens2;
  m_noutputs = noutputs;
  if (m_nbins > 1)
    {
      m_noutputs = noutputs * m_nbins; // we several outputs for each actuator
      m_noutputsbins = noutputs;        // we store the number of actuators
    }
  m_nneurons = (m_ninputs + m_nhiddens + m_noutputs);
  m_nlayers = nlayers;
  if (m_nhiddens == 0)
   m_nlayers = 0;
  m_bias = bias;
  m_netType = netType;
  m_actFunct = actFunct;
  m_outType = outType;
  m_wInit = wInit;
  m_clip = clip;
  m_normalize = normalize;
  m_randAct = randAct;
  m_randActR = randActR;
  m_low = low;
  m_high = high;
  m_wrange = wrange;
  if (m_netType > 0 && m_nlayers > 1)
       {
        printf("WARNING: NUMBER OF LAYERS FORCED TO 1 SINCE ONLY FEED_FORWARD NETWORKS CAN HAVE MULTIPLE LAYERS");
        m_nlayers = 1;
       }
  // LSTM networks should use fake linear functions for internal neurons
  if (m_netType == 3)
        m_actFunct = 3;
  netRng = new RandomGenerator(time(NULL));

  // display info and check parameters are in range
  printf("Network %d->", m_ninputs);
  int l;
  if (m_nhiddens2 == 0 || m_nlayers != 2)
    {
    for(l=0; l < m_nlayers; l++)
      printf("%d->", m_nhiddens / m_nlayers);
    }
    else
    {
      printf("%d->", m_nhiddens - m_nhiddens2);
      printf("%d->", m_nhiddens2);
    }
  printf("%d ", m_noutputs);
  if (m_netType == 0)
    printf("feedforward ");
  else if (m_netType == 1)
    printf("recurrent ");
  else if (m_netType == 2)
    printf("fully recurrent ");
  else if (m_netType == 3)
    printf("LSTM ");
  if (m_bias)
    printf("with bias ");
  switch (m_actFunct)
    {
      case 1:
        printf("logistic ");
        break;
      case 2:
        printf("tanh ");
        break;
      case 3:
        printf("linear ");
        break;
      case 4:
        printf("binary ");
        break;
    }
  switch (m_outType)
    {
       case 1:
         printf("output:logistic ");
         break;
       case 2:
         printf("output:tanh ");
         break;
       case 3:
         printf("output:linear ");
         break;
       case 4:
         printf("output:binary ");
         break;
    }
  if (m_nbins > 1)
    printf("bins: %d", m_nbins);
  if (m_wInit < 0 || m_wInit > 2) 
    m_wInit = 0;
  switch (m_wInit)
    {
       case 0:
         printf("init:xavier ");
         break;
       case 1:
         printf("init: norm-incoming ");
         break;
       case 2:
         printf("init: uniform ");
         break;
    }
  if (m_normalize < 0 || m_normalize > 1)
    m_normalize = 0;
  if (m_normalize == 1)
    printf("input-normalization ");
  if (m_clip < 0 || m_clip > 1)
    m_clip = 0;
  if (m_clip == 1)
    printf("clip ");
  if (m_randAct < 0 || m_randAct > 2)
    m_randAct = 0;
  switch (m_randAct)
    {
       case 1:
         printf("motor-noise %.2f ", m_randActR);
         break;
       case 2:
         printf("diagonal gaussian ");
         break;
    }  
  printf("\n");
    
  // allocate variables
  m_nblocks = 0;
  m_netinput = new double[m_nneurons]; // DEBUG ALREADY ALLOCATE IN THE FUNCTION ABOVE

  // states required for LSTM networks
  if (m_netType == 3)
    {
      m_nstate = new double[m_nneurons * m_nnetworks];
      m_pnstate = new double[m_nneurons * m_nnetworks];
    }
  m_netblock = new int[MAX_BLOCKS * 5];  // DEBUG ALREADY ALLOCATED IN THE FUNCTION ABOVE
  m_neurontype = new int[m_nneurons];
 
  initNetArchitecture(); // Initialize the blocks that describe the architecture of the network
 
  
  if (normalize == 1)  // Initialize and allocate vector for observation normalization
    {
      m_mean = new double[m_ninputs];  // mean used for normalize
      m_std = new double[m_ninputs];   // standard deviation used for normalize
      m_sum = new double[m_ninputs];   // sum of input data used for calculating normalization vectors
      m_sumsq = new double[m_ninputs]; // squared sum of input data used for calculating normalization vectors
      m_count = 0.01;                  // to avoid division by 0
      int i;
      for (i = 0; i < m_ninputs; i++)
        {
          m_sum[i] = 0.0;
          m_sumsq[i] = 0.01;    
          m_mean[i] = 0.0;
          m_std[i] = 1.0;
        }
    }
}


// destructor, yet to be implemented
Evonet::~Evonet()
{
}


// initialize the neurontype vector and
// create the block that describe the architecture of the policy
// and initialize thevector that specify the activation function of the neurons
void Evonet::initNetArchitecture()
{
  int* nbl;
  int* nt;
  int n;

  // neurons' type
  for (n = 0, nt = m_neurontype; n < m_nneurons; n++, nt++)
    {
      if (n < m_ninputs)
        *nt = 0; // inputs neurons are of type 0
      else
        {
          if (n < (m_ninputs + m_nhiddens))
             *nt = m_actFunct; // hidden neurons type is set on the basis of the m_actFunct user's parameter 
          else
             *nt = m_outType;  // output neurons type is set on the basis of the m_outType user's parameter 
        }
    }

  m_nblocks = 0;
  nbl = m_netblock;
    
  // in all cases the first block is an update block used to update the sensory neurons
  *nbl = 1; nbl++;
  *nbl = 0; nbl++;
  *nbl = m_ninputs; nbl++;
  *nbl = 0; nbl++;
  *nbl = 0; nbl++;
  m_nblocks++;
  
  // feed-forward architecture with a variable number of internal layers
  if (m_netType == 0)
    {
      if (m_nlayers == 0)  // perceptron with sensory neurons connected directly to motor neurons
        {
          // connection block in which motors neurons receive connections directly from sensors
          *nbl = 0; nbl++;
          *nbl = m_ninputs; nbl++;
          *nbl = m_noutputs; nbl++;
          *nbl = 0; nbl++;
          *nbl = m_ninputs; nbl++;
          m_nblocks++;

         // update block to be used to update motor neurons
         *nbl = 1; nbl++;
         *nbl = m_ninputs; nbl++;
         *nbl = m_noutputs; nbl++;
         *nbl = 0; nbl++;
         *nbl = 0; nbl++;
         m_nblocks++;
        }
      else
        {
          int ninternals;
          ninternals = m_nhiddens / m_nlayers;
          int l;
          int idpresynaptic;
          int npresynaptic;
          int idpostsynaptic = 0;
          int npostsynaptic = m_ninputs;
          // input-hidden connection
          for(l=0; l < m_nlayers; l++)
            {
              if (m_nlayers == 2 && m_nhiddens2 > 0)
               {
                if (l == 0)
                  {
                    ninternals = m_nhiddens - m_nhiddens2;
                  }
                else
                  {
                    ninternals = m_nhiddens2;
                  }
               }
              // updte neurons index
              idpresynaptic = idpostsynaptic;
              npresynaptic = npostsynaptic;
              idpostsynaptic = idpresynaptic + npresynaptic;
              npostsynaptic = ninternals;        
              // connection block
              *nbl = 0; nbl++;
              *nbl = idpostsynaptic; nbl++;
              *nbl = npostsynaptic; nbl++;
              *nbl = idpresynaptic; nbl++;
              *nbl = npresynaptic; nbl++;
              m_nblocks++;   
              // updated block
              *nbl = 1; nbl++;
              *nbl = idpostsynaptic; nbl++;
              *nbl = npostsynaptic; nbl++;
              *nbl = 0; nbl++;
              *nbl = 0; nbl++;
              m_nblocks++;     
            }
           // update neurons index
           idpresynaptic = idpostsynaptic;
           npresynaptic = npostsynaptic;
           idpostsynaptic = idpresynaptic + npresynaptic;
           npostsynaptic = m_noutputs;              
           // connection block to the motor neurons
           *nbl = 0; nbl++;
           *nbl = idpostsynaptic; nbl++;
           *nbl = npostsynaptic; nbl++;
           *nbl = idpresynaptic; nbl++;
           *nbl = npresynaptic; nbl++;
           m_nblocks++;           
           // updated block
           *nbl = 1; nbl++;
           *nbl = idpostsynaptic; nbl++;
           *nbl = npostsynaptic; nbl++;
           *nbl = 0; nbl++;
           *nbl = 0; nbl++;
           m_nblocks++;
         }
     }

  // network with 1 internal layer with recurrent connections
  if (m_netType == 1)
    {
      // input-hidden connections
      *nbl = 0; nbl++;
      *nbl = m_ninputs; nbl++;
      *nbl = m_nhiddens; nbl++;
      *nbl = 0; nbl++;
      *nbl = m_ninputs; nbl++;
      m_nblocks++;
    
      // hidden-hidden connections
      *nbl = 0; nbl++;
      *nbl = m_ninputs; nbl++;
      *nbl = m_nhiddens; nbl++;
      *nbl = m_ninputs; nbl++;
      *nbl = m_nhiddens; nbl++;
      m_nblocks++;
    
      // hidden update block
      *nbl = 1; nbl++;
      *nbl = m_ninputs; nbl++;
      *nbl = m_nhiddens; nbl++;
      *nbl = 0; nbl++;
      *nbl = 0; nbl++;
      m_nblocks++;

      // hidden-output connections
      *nbl = 0; nbl++;
      *nbl = m_ninputs + m_nhiddens; nbl++;
      *nbl = m_noutputs; nbl++;
      *nbl = m_ninputs; nbl++;
      *nbl = m_nhiddens; nbl++;
      m_nblocks++;
      
      // output-output connections
      *nbl = 0; nbl++;
      *nbl = m_ninputs + m_nhiddens; nbl++;
      *nbl = m_noutputs; nbl++;
      *nbl = m_ninputs + m_nhiddens; nbl++;
      *nbl = m_noutputs; nbl++;
      m_nblocks++;
    
      // output update block
      *nbl = 1; nbl++;
      *nbl = m_ninputs + m_nhiddens; nbl++;
      *nbl = m_noutputs; nbl++;
      *nbl = 0; nbl++;
      *nbl = 0; nbl++;
      m_nblocks++;
 }
 
  // Fully-recurrent network (hiddens and outputs receive connections from input, hiddens and outputs)
  if (m_netType == 2)
   {
    *nbl = 0; nbl++;
    *nbl = m_ninputs; nbl++;
    *nbl = m_nhiddens + m_noutputs; nbl++;
    *nbl = 0; nbl++;
    *nbl = m_ninputs + m_nhiddens + m_noutputs; nbl++;
    m_nblocks++;
  
    // hidden-output update block
    *nbl = 1; nbl++;
    *nbl = m_ninputs; nbl++;
    *nbl = m_nhiddens + m_noutputs; nbl++;
    *nbl = 0; nbl++;
    *nbl = 0; nbl++;
    m_nblocks++;
 }
  
  // recurrent LSTM, 1 layer
  if (m_netType == 3)
    {        
      // hiddens receive connections from input and hiddens
      *nbl = 2; nbl++;  // special connection block, type 2
      *nbl = m_ninputs; nbl++;
      *nbl = m_nhiddens; nbl++;
      *nbl = 0; nbl++;
      *nbl = m_ninputs + m_nhiddens; nbl++;
      m_nblocks++;
        
      // hidden update block
      *nbl = 1; nbl++;
      *nbl = m_ninputs; nbl++;
      *nbl = m_nhiddens; nbl++;
      *nbl = 0; nbl++;
      *nbl = 0; nbl++;
      m_nblocks++;
        
      // hidden-output connections
      *nbl = 0; nbl++;
      *nbl = m_ninputs + m_nhiddens; nbl++;
      *nbl = m_noutputs; nbl++;
      *nbl = m_ninputs; nbl++;
      *nbl = m_nhiddens; nbl++;
      m_nblocks++;
        
      // output update block
      *nbl = 1; nbl++;
      *nbl = m_ninputs + m_nhiddens; nbl++;
      *nbl = m_noutputs; nbl++;
      *nbl = 0; nbl++;
      *nbl = 0; nbl++;
      m_nblocks++;        
    }
  
  // display connection blocks (when verbose is set to 1)
  int b;
  if (verbose == 1)
  {
    printf("the policy will be updated as follow:\n");
    for (b = 0, nbl = m_netblock; b < m_nblocks; b++, nbl = (nbl + 5))
    {
      if (*nbl == 0)
        printf("netinput of neurons [%d-%d] computed on the basis of activation received from neurons [%d-%d] \n", *(nbl + 1), *(nbl + 1) + *(nbl + 2),*(nbl + 3),*(nbl + 3) + *(nbl + 4));
      if (*nbl == 1)
        printf("neuron [%d-%d] updated\n", *(nbl + 1), *(nbl + 1) + *(nbl + 2));
      if (*nbl == 2)
        printf("netinput of LSTM units [%d-%d] computed on the basis of activation received from neurons [%d-%d] \n", *(nbl + 1), *(nbl + 1) + *(nbl + 2),*(nbl + 3),*(nbl + 3) + *(nbl + 4));
    }
  }
    
}



// reset the activation of the neurons 
void Evonet::resetNet()
{
  int i;
  int n;
  double *neura;
    
  neura = neuronact;
  for (n = 0; n < m_nnetworks; n++)
    {
      for (i = 0; i < m_nneurons; i++, neura++)
        *neura = 0.0;
    }
  if (m_netType == 3)
     {
       double *neurs;
       neurs = m_nstate;
       for (n = 0; n < m_nnetworks; n++)
         {
            for (i = 0; i < m_nneurons; i++, neurs++)
              *neurs = 0.0;
         }
     }
}



// update net
void Evonet::updateNet()
{
  double* p;         // free parameters
  double* neurona;   // the activation vector of the current network
  float* cobserv;    // the observation vector of the current network
  float* cact;       // the action vector of the current network
  double* nstate;    // the lstm-state vector of the current network
  double* pnstate;  // the previous lstm-state of the current network
  double* a;
  double* ni;
  int i;
  int t;
  int b;
  int* nbl;
  int* nt;
  int j;
  int n;
  double lstm[4]; // gates: forget, input, output, gate-gate
 
 
  if (m_heterogeneous == 1)
    p = cgenotype;

  // for each network
  for(n=0, neurona = neuronact, cobserv = cobservation, cact = caction, nstate = m_nstate, pnstate = m_pnstate; n < m_nnetworks; n++, neurona = (neurona + m_nneurons), cobserv = (cobserv + m_ninputs), cact = (cact + m_noutputs))
    {
      // in case of homogeneous networks we use the same parameters for multiple networks
      if (m_heterogeneous == 0)
        p = cgenotype;
        
      // Collect the input for updatig the normalization
      // We do that before eventually clipping (not clear whether this is the best choice)
      if (m_normalize  == 1 && m_normphase == 1)
        collectNormalizationData();
        
      // Normalize input
      if (m_normalize == 1)
        {
          for (j = 0; j < m_ninputs; j++)
            cobserv[j] = (cobserv[j] - m_mean[j]) / m_std[j];
        }
        
      // Clip input values
      if (m_clip == 1)
        {
          for (j = 0; j < m_ninputs; j++)
            {
              if (cobserv[j] < -CLIP_VALUE)
                 cobserv[j] = -CLIP_VALUE;
              if (cobserv[j] > CLIP_VALUE)
                  cobserv[j] = CLIP_VALUE;
            }
        }
        
      // compute biases (the netinput of each neuron is initialized with the bias
      if (m_bias == 1)
        {
          // Only non-input neurons have bias
          for(i = 0, ni = m_netinput; i < m_nneurons; i++, ni++)
            {
              if (i >= m_ninputs)
                {
                  *ni = *p;
                  p++;
                }
                else
                {
                  *ni = 0.0;
                }
            }
        }
        
      // for each tuple block
      for (b = 0, nbl = m_netblock; b < m_nblocks; b++)
        {
          // connection block, the netinput is the dot product of the activation of the presynaptic neurons for the corresponding connectio weights
          if (*nbl == 0)
            {
              for(t = 0, ni = (m_netinput + *(nbl + 1)); t < *(nbl + 2); t++, ni++)
                {
                   for(i = 0, a = (neurona + *(nbl + 3)); i < *(nbl + 4);i++, a++)
                    {
                      *ni += *a * *p;
                      p++;
                    }
                }
            }            
          // LSTM connection block, in this case each unit include 3 gating neurons and a memory neuron
          // in this case we compute the netinput and then the activation of the four neurons and we store the output of the cell in the netinput of the unit
          if (*nbl == 2)
            {
              int tt;
              for(t = 0, tt = *(nbl + 1); t < *(nbl + 2); t++, tt++)
                pnstate[tt] = nstate[tt];
              for(t = 0, ni = (m_netinput + *(nbl + 1)), tt = *(nbl + 1); t < *(nbl + 2); t++, ni++, tt++)
                {
                  for (int g = 0; g < 4; g++)
                    {
                      lstm[g] = 0;
                      for(i = 0, a = (neurona + *(nbl + 3)); i < *(nbl + 4); i++, a++)
                        {
                          lstm[g] += *a * *p;
                          p++;
                        }
                      // forget state can be sigmoid(netinput+1.0) where 1.0 is forget_bias
                      switch(g)
                       {
                         case 0: // forget gate
                           lstm[0] = logistic(lstm[g]+1.0); // +1.0 implement the forget bias
                           break;
                         case 1: // input gate
                           lstm[1] = logistic(lstm[g]);
                           break;
                         case 2: // output gate
                           lstm[2] = logistic(lstm[g]);
                           break;
                         case 3: // act
                           lstm[3] = tanh(lstm[g]);
                           break;
                       }
                    }
                  // state = previous_state * forget_gate + gate_act * input_gate
                  nstate[tt] = pnstate[tt] * lstm[0] + lstm[3] * lstm[1];
                  // output = state * output_gate
                  // we store it in the netinput, it will be then copied in the activation by the next update block through the usage of a linear activation function
                  *ni = tanh(nstate[tt]) * lstm[2];
                }
            }
     
          // update block, update the activation of the neurons of the block, the activation is a function of the netinput
          if (*nbl == 1)
            {
              for(t = *(nbl + 1), a = (neurona + *(nbl + 1)), ni = (m_netinput + *(nbl + 1)), nt = (m_neurontype + *(nbl + 1)); t < (*(nbl + 1) + *(nbl + 2)); t++, a++, ni++, nt++)
                {
                  switch (*nt)
                    {
                      case 0:
                        // the activation of input neuron is just copied from the observation vector
                        *a = *(cobserv + t);
                        break;
                      case 1:
                        // logistic neuron
                        *a = logistic(*ni);
                        break;
                      case 2:
                        // hyperbolic tangent neuron
                        *a = tanh(*ni);
                        break;
                      case 3:
                        // linear neuron
                        *a = linear(*ni);
                        break;
                      case 4:
                        // binary neurons
                        if (*ni >= 0.5)
                           *a = 1.0;
                         else
                           *a = -1.0;
                         break;
                    }
                }
            }
            nbl = (nbl + 5);
        }
        // copy the activation of motor neurons in the action vector
        // we might also perturb the state of motor neurons with noise
        if (m_nbins == 1) // we are not using bins here
        {
          int i;
          for (i = 0; i < m_noutputs; i++)
            {
              switch(m_randAct)
                {
                  // no noise
                  case 0:
                    cact[i] = neurona[m_ninputs + m_nhiddens + i];
                    break;
                  // gaussian noise with fixed range
                  case 1:
                    cact[i] = neurona[m_ninputs + m_nhiddens + i] + (netRng->getGaussian(1.0, 0.0) * m_randActR);
                    break;
                  // gaussian noise with parametric range (diagonal-gaussian), the range of noise depends on parameters adapted together with the connection weights
                  case 2:
                    cact[i] = neurona[m_ninputs + m_nhiddens + i] + (netRng->getGaussian(1.0, 0.0) * exp(*p));
                    p++;
                    break;
                }
            }
        }
        else // we use N motor neurons to encode N possible values that each motor neuron can assume. The value will correspond to that of the neuron more activated
        {
            int i = 0;
            int j = 0;
            double ccact;
            int ccidx;
            // For each output, we look for the bin with the highest activation
            for (i = 0; i < m_noutputsbins; i++)
            {
                // Current highest activation
                ccact = -9999.0;
                // Index of the current highest activation
                ccidx = -1;
                for (j = 0; j < m_nbins; j++)
                {
                    if (neurona[m_ninputs + m_nhiddens + ((i * m_nbins) + j)] > ccact)
                    {
                        ccact = neurona[m_ninputs + m_nhiddens + ((i * m_nbins) + j)];
                        ccidx = j;
                    }
                }
                cact[i] = 1.0 / ((double)m_nbins - 1.0) * (double)ccidx * (m_high - m_low) + m_low;
                if (m_randAct == 1)
                    cact[i] += (netRng->getGaussian(1.0, 0.0) * 0.01);
            }
        }
  
    // we advance the pointers of the neuron-state variable (only in case of SLTM networks)
    if (m_netType == 3)
       {
          nstate = (nstate + m_nneurons);
          pnstate = (pnstate + m_nneurons);
       }
  
    }
    
}



// compute the number of parameters required by the network
// i.e. the number of conection weights, biases, and eventually action perturbation parameters
int Evonet::computeParameters()
{
  int nparams;
  int* nbl;
  int b;

  nparams = 0;
 
  // biases
  if (m_bias)
    nparams += (m_nhiddens + m_noutputs);
    
  // blocks
  for (b = 0, nbl = m_netblock; b < m_nblocks; b++)
    {
      if (*nbl == 0)
        nparams += *(nbl + 2) * *(nbl + 4);  // connection block
      if (*nbl == 2)
        nparams += *(nbl + 2) * *(nbl + 4) * 4; // LSTM connection block
      nbl = (nbl + 5);
    }
 
  // parametrized gaussian noise on action 
  if (m_randAct == 2)
    nparams += m_noutputs;

  // heterogenoeus network have different parameters for each network
  if (m_heterogeneous == 1)
    nparams *= m_nnetworks;
    
  m_nparams = nparams;

  return nparams;
}


// initialize weights
void Evonet::initWeights()
{
  int i;
  int j;
  int t;
  int b;
  int n;
  int* nbl;
  double range;
  int nnetworks;
  bool LSTM;
    
  // in case of homogeneous networks we need a single set of parameter
  if (m_heterogeneous == 0)
     nnetworks = 1;
    else
     nnetworks = m_nnetworks;
  
  // for each network
  j = 0;
  for(n=0; n < nnetworks; n++)
    {
    // Initialize biases to 0.0
    if (m_bias)
    {
        for (i = 0; i < (m_nhiddens + m_noutputs); i++)
        {
            cgenotype[j] = 0.0;
            j++;
        }
    }
    // Initialize weights of connection blocks
    for (b = 0, nbl = m_netblock; b < m_nblocks; b++)
    {
      // connection block (2 is for LSTM)
      if (*nbl == 0 || *nbl == 2)
        {
           if (*nbl == 2)
             LSTM = true;
           else
             LSTM = false;
           
           switch (m_wInit)
            {
              // xavier initialization, i.e. gaussian distribution with range = (radq(2.0 / (npresynaptic + npostsynaptic))
              case 0:
                int nin;
                int nout;
                // ninput and noutput of the current block
                nin = *(nbl + 4);
                nout = *(nbl + 2);
                // if previous and/or next block include the same receiving neurons we increase ninputs accordingly                                                                                      
                // connection block are always preceded by update block, so we can check previous values
                if ((*(nbl + 5) == 0) && ((*(nbl + 1) == *(nbl + 6)) && (*(nbl + 2) == *(nbl + 7))))
                  nin += *(nbl + 9);
                if ((*(nbl - 5) == 0) && ((*(nbl - 4) == *(nbl + 1)) && (*(nbl - 3) == *(nbl + 2))))
                  nin += *(nbl - 1);
                // compute xavier range
                range = sqrt(2.0 / (nin + nout));
                for (t = 0; t < *(nbl + 2); t++)
                  {
                    for (i = 0; i < *(nbl + 4); i++)
                      {
                         if (LSTM)
                            {
                               for (int ii=0; ii < 4; ii++)
                                 {
                                   cgenotype[j] = netRng->getGaussian(range, 0.0);
                                   j++;
                                 }
                            }
                          else
                            {
                              cgenotype[j] = netRng->getGaussian(range, 0.0);
                              j++;
                            }
                        }
                    }
                break;
              // normalization of incoming weights as in salimans et al. (2017)
              // in case of linear output, use a smaller range for the last layer
              // we assume that the last layer corresponds to the last connection block followed by the last update block
              // compute the squared sum of gaussian numbers in order to scale the weights
              // equivalent to the following python code for tensorflow:
              // out = np.random.randn(*shape).astype(np.double32)
              // out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
              // where randn extract samples from Gaussian distribution with mean 0.0 and std 1.0
              // std is either 1.0 or 0.01 depending on the layer
              // np.square(out).sum(axis=0, keepdims=True) computes the squared sum of the elements in out
              // DUBUG, TO BE ADAPTED FOR LSTM                                                                                      
              case 1:
                {
                  double *wSqSum = new double[*(nbl + 2)];
                  int k;
                  int cnt;
                  range = 1.0; // std
                  if (m_outType == 3 && b == (m_nblocks - 2))
                    range = 0.01; // std for layers followed by linear outputs
                  for (t = 0; t < *(nbl + 2); t++)
                    wSqSum[t] = 0.0;
                  // Index storing the genotype block to be normalized (i.e., the starting index)
                  k = j;
                  // Counter of weights
                  cnt = 0;
                  for (t = 0; t < *(nbl + 2); t++)
                    {
                       for (i = 0; i < *(nbl + 4); i++)
                         {
                            // Extract weights from Gaussian distribution with mean 0.0 and std 1.0
                            cgenotype[j] = netRng->getGaussian(1.0, 0.0);
                            // Update square sum of weights
                            wSqSum[t] += (cgenotype[j] * cgenotype[j]);
                            j++;
                            // Update counter of weights
                            cnt++;
                         }
                    }
                  // Normalize weights
                  j = k;
                  t = 0;
                  i = 0;
                  while (j < (k + cnt))
                    {
                       cgenotype[j] *= (range / sqrt(wSqSum[t])); // Normalization factor
                       j++;
                       i++;
                       if (i % *(nbl + 4) == 0)
                          // Move to next sum
                          t++;
                    }
                  // We delete the pointer
                  delete[] wSqSum;
                  }
                break;
              // normal gaussian distribution with range netWrange
              case 2:
                // the range is specified manually and is the same for all layers
                for (t = 0; t < *(nbl + 2); t++)
                  {
                    for (i = 0; i < *(nbl + 4); i++)
                      {
                        if (LSTM)
                          {
                            for (int ii=0; ii < 4; ii++)
                              {
                                cgenotype[j] = netRng->getGaussian(range, 0.0);
                                j++;
                              }
                          }
                        else
                          {
                            cgenotype[j] = netRng->getDouble(-m_wrange, m_wrange);
                            j++;
                          }
                      }
                  }
                break;
                default:
                    // unrecognized initialization mode
                    printf("ERROR: unrecognized initialization mode: %d \n", m_wInit);
                break;
            }
        }
        nbl = (nbl + 5);
    }
    // parameters for the diagonal gaussian output
    if (m_randAct == 2)
      {
        for (i=0; i < m_noutputs; i++)
          {
            cgenotype[j] = 0.0;
            j++;
          }
      }

    // uncomment the following to check that the sum of absolute incoming weights is appropriately normalized
    /* print sum of absolute incoming weight
    j = 0;
    if (m_bias)
      {
         for (i = 0; i < (m_nhiddens + m_noutputs); i++)
            j++;
      }
    double sum;
    for (b = 0, nbl = m_netblock; b < m_nblocks; b++)
      {
       printf("block %d type %d\n", b, *nbl);
       if (*nbl == 0)
         {
           for(t = 0; t < *(nbl + 2); t++)
            {
              sum = 0;
              for(i = 0; i < *(nbl + 4); i++)
                {
                  sum += fabs(cgenotype[j]);
                  j++;
                }
            printf("block %d neuron %d sum-abs incoming weights %f\n", b, t, sum);
         }
      }
      nbl = (nbl + 5);
    }
    */
      
    }
}


// set the normalization phase (0 = do nothing, 1 = collect data to update normalization vectors)
void Evonet::normphase(int phase)
{
   m_normphase = phase;
}

// collect data for normalization
void Evonet::collectNormalizationData()
{
  int i;

  // DEBUG, THE NEXT UNCOMMENTED FOR AND THE NEXT LINE SHOULD BE REMOVED
  for (i = 0; i < m_ninputs; i++)
       //printf("%.2f ", cobservation[i]);

  for (i = 0; i < m_ninputs; i++)
   {
     m_sum[i] += cobservation[i];
     m_sumsq[i] += (cobservation[i] * cobservation[i]);
   }

  // Update counter
  m_count++;
}

// compute normalization vectors
void Evonet::updateNormalizationVectors()
{
  int i;
  int ii;
  double cStd;
 
  for (i = 0; i < m_ninputs; i++)
    {
      m_mean[i] = m_sum[i] / m_count;
      cStd = (m_sumsq[i] / m_count - (m_mean[i] * m_mean[i]));
      if (cStd < 0.01)
        cStd = 0.01;
      m_std[i] = sqrt(cStd);
    }
  // copy nornalization vectors on the cnormalization vector that is used for saving data
  ii = 0;
  for (i = 0; i < m_ninputs; i++)
    {
       cnormalization[ii] = m_mean[i];
       ii++;
    }
 for (i = 0; i < m_ninputs; i++)
    {
       cnormalization[ii] = m_std[i];
       ii++;
    }
}

// restore a loaded normalization vector
void Evonet::setNormalizationVectors()
{

  int i;
  int ii;
 
  if (m_normalize == 1)
  {
    ii = 0;
    for (i = 0; i < m_ninputs; i++)
      {
         m_mean[i] = cnormalization[ii];
         ii++;
      }
    for (i = 0; i < m_ninputs; i++)
      {
         m_std[i] = cnormalization[ii];
         ii++;
      }
   }
}

// reset normalization vector
void Evonet::resetNormalizationVectors()
{

  if (m_normalize == 1)
    {
      m_count = 0.01; // to avoid division by 0
      int i;
      for (i = 0; i < m_ninputs; i++)
        {
          m_sum[i] = 0.0;
          m_sumsq[i] = 0.01; // eps
          m_mean[i] = 0.0;
          m_std[i] = 1.0;
        }
    }
}

// get the current normalization vector (copy from m_mean[] and m_std[] to cnormalization)
void Evonet::getNormalizationVectors()
{
    
    int i;
    int ii;
    
    if (m_normalize == 1)
    {
        ii = 0;
        for (i = 0; i < m_ninputs; i++)
        {
            cnormalization[ii] = m_mean[i];
            ii++;
        }
        for (i = 0; i < m_ninputs; i++)
        {
            cnormalization[ii] = m_std[i];
            ii++;
        }
    }
}

// store the pointer to the parameter vector
void Evonet::copyGenotype(double* genotype)
{
  cgenotype = genotype;
}

// store the pointer to the observation vector
void Evonet::copyInput(float* input)
{
  cobservation = input;
}

// store the pointer to the action vector
void Evonet::copyOutput(float* output)
{
  caction = output;
}

// store the pointer to the neuron activation vector
void Evonet::copyNeuronact(double* na)
{
  neuronact = na;
}

// store the pointer to the normalization vector
void Evonet::copyNormalization(double* no)
{
    cnormalization = no;
}

// useless function
void Evonet::initNet(char* filename)
{
    
}
