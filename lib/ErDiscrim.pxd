"""
This file belong to https://github.com/snolfi/evorobotpy
Author: Stefano Nolfi, stefano.nolfi@istc.cnr.it

ErDisrim.pxd, python wrapper for discrim.cpp

This file is part of the python module ErDiscrim.so that include the following files:
discrim.cpp, discrim.h, robot-env.cpp, robot-env.h, utilities.cpp, utilities.h, ErDiscrim.pxd, ErDiscrim.pyx and setupErDiscrim.py
And can be compiled with cython and installed with the commands: cd ./evorobotpy/lib; python3 setupErDiscrim.py build_ext –inplace; cp ErDiscrim*.so ../bin
"""

cdef extern from "utilities.cpp":
    pass

cdef extern from "discrim.cpp":
    pass

# Declare the class with cdef
#cdef extern from "utilities.h":
    #cdef cppclass RandomGenerator:
        #RandomGenerator() except +
        #void setSeed(int seed)
        #int seed()
        #int getInt(int min, int max)
        #double getDouble(double min, double max)
        #double getGaussian(double var, double mean)

# Declare the class with cdef
cdef extern from "discrim.h":
    cdef cppclass Problem:
        Problem() except +
        int m_trial
        int ninputs
        int noutputs
        double* m_state
        double m_masspole_2, m_length_2
        void seed(int s)
        void reset()
        double step()
        void close()
        void render()
        double isDone()
        void copyObs(float* observation)
        void copyAct(float* action)
        void copyDone(int* done)
        void copyDobj(double* objs)

