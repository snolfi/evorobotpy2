"""
This file belong to https://github.com/snolfi/evorobotpy
Author: Stefano Nolfi, stefano.nolfi@istc.cnr.it

ErStaybehind.pxd, python wrapper for predprey.cpp

This file is part of the python module ErPredprey.so that include the following files:
staybehind.cpp, staybehind.h, robot-env.cpp, robot-env.h, utilities.cpp, utilities.h, ErPredprey.pxd, ErPredprey.pyx and setupErPredprey.py
And can be compiled with cython and installed with the commands: cd ./evorobotpy/lib; python3 setupErPredprey.py build_ext –inplace; cp ErPredprey*.so ../bin
"""

cdef extern from "utilities.cpp":
    pass

cdef extern from "staybehind.cpp":
    pass

# Declare the class with cdef
cdef extern from "predprey.h":
    cdef cppclass Problem:
        Problem() except +
        int ninputs
        int noutputs
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

