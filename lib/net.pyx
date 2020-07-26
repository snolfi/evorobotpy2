"""
This file belong to https://github.com/snolfi/evorobotpy
Author: Stefano Nolfi, stefano.nolfi@istc.cnr.it

net.pyx, python wrapper for evonet.cpp

This file is part of the python module net.so that include the following files:
evonet.cpp, evonet.h, utilities.cpp, utilities.h, net.pxd, net.pyx and setupevonet.py
And can be compile with cython with the commands: cd ./evorobotpy/lib; python3 setupevonet.py build_ext –inplace; cp net*.so ../bin
"""

# distutils: language=c++

import cython
# import both numpy and the Cython declarations for numpy
import numpy as np
import time
cimport numpy as np
from libcpp cimport bool
from net cimport Evonet

# PyEvonet
cdef class PyEvonet:
    cdef Evonet c_net

    def __cinit__(self):
        self.c_net = Evonet()

    def __cinit__(self, int nnetworks, int heterogeneous, int ninputs, int nhiddens, int noutputs, int nlayers, int nhiddens2, int bias, int netType, int actFunct, int outType, int wInit, int clip, int normalize, int randAct, double randActR, double wrange, int nbins, double low, double high):
        self.c_net = Evonet(nnetworks, heterogeneous, ninputs, nhiddens, noutputs, nlayers, nhiddens2, bias, netType, actFunct, outType, wInit, clip, normalize, randAct, randActR, wrange, nbins, low, high)

    def seed(self, int s):
        self.c_net.seed(s)   

    def resetNet(self):
        self.c_net.resetNet()

    def copyGenotype(self, np.ndarray[double, ndim=1, mode="c"] geno not None):
        self.c_net.copyGenotype(&geno[0])

    def copyInput(self, np.ndarray[float, ndim=1, mode="c"] inp not None):
        self.c_net.copyInput(&inp[0])

    def copyOutput(self, np.ndarray[float, ndim=1, mode="c"] outp not None):
        self.c_net.copyOutput(&outp[0])

    def copyNeuronact(self, np.ndarray[double, ndim=1, mode="c"] na not None):
        self.c_net.copyNeuronact(&na[0])

    def copyNormalization(self, np.ndarray[double, ndim=1, mode="c"] no not None):
        self.c_net.copyNormalization(&no[0])

    def updateNet(self):
        self.c_net.updateNet()

    def computeParameters(self):
        return self.c_net.computeParameters()

    def initWeights(self):
        self.c_net.initWeights()

    def normphase(self, int phase):
        self.c_net.normphase(phase)

    def updateNormalizationVectors(self):
        self.c_net.updateNormalizationVectors()

    def setNormalizationVectors(self):
        self.c_net.setNormalizationVectors()

    def getNormalizationVectors(self):
        self.c_net.getNormalizationVectors()

    def resetNormalizationVectors(self):
        self.c_net.resetNormalizationVectors()

    # Attribute access
    #@property
    #def ninputs(self):
        #return self.c_net.m_ninputs

    # Attribute access
    #@property
    #def noutputs(self):
        #return self.c_net.m_noutputs

    # Attribute access
    #@property
    #def nhiddens(self):
        #return self.c_net.m_nhiddens

