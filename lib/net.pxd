"""
This file belong to https://github.com/snolfi/evorobotpy
Author: Stefano Nolfi, stefano.nolfi@istc.cnr.it

net.pxd, python wrapper for evonet.cpp

This file is part of the python module net.so that include the following files:
evonet.cpp, evonet.h, utilities.cpp, utilities.h, net.pxd, net.pyx and setupevonet.py
And can be compile with cython with the commands: cd ./evorobotpy/lib; python3 setupevonet.py build_ext –inplace; cp net*.so ../bin
"""

cdef extern from "utilities.cpp":
    pass

cdef extern from "evonet.cpp":
    pass

# Declare the class with cdef
cdef extern from "evonet.h":
    cdef cppclass Evonet:
        Evonet() except +
        Evonet(int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, double, double, int, double, double) except +
        int nnetworks, heterogeneous, ninputs, nhiddens, noutputs, nlayers, nhiddens2, bias, netType, actFunct, outType, wInit, clip, normalize, randAct, randActR, wrange, nbins, low, high
        double* m_act
        double* m_netinput
        void resetNet()
        void seed(int s)
        void copyGenotype(double* genotype)
        void copyInput(float* input)
        void copyOutput(float* output)
        void copyNeuronact(double* na)
        void copyNormalization(double* no)
        void updateNet()
        void getOutput(double* output)
        int computeParameters()
        void initWeights()
        void normphase(int phase)
        void updateNormalizationVectors()
        void setNormalizationVectors()
        void getNormalizationVectors()
        void resetNormalizationVectors()

