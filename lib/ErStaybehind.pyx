"""
This file belong to https://github.com/snolfi/evorobotpy
Author: Stefano Nolfi, stefano.nolfi@istc.cnr.it

ErStaybehind.pyx, python wrapper for staybehind.cpp

This file is part of the python module ErPredprey.so that include the following files:
staybehind.cpp, staybehind.h, robot-env.cpp, robot-env.h, utilities.cpp, utilities.h, ErPredprey.pxd, ErPredprey.pyx and setupErPredprey.py
And can be compiled with cython and installed with the commands: cd ./evorobotpy/lib; python3 setupErPredprey.py build_ext –inplace; cp ErPredprey*.so ../bin
"""

# distutils: language=c++

import cython
# import both numpy and the Cython declarations for numpy
import numpy as np
import time
cimport numpy as np
from libcpp cimport bool
from ErStaybehind cimport Problem

# PyProblem
cdef class PyErProblem:
    cdef Problem cproblem

    def __cinit__(self):
        self.cproblem = Problem()

    def seed(self, int s):
        self.cproblem.seed(s)   

    def reset(self):
        self.cproblem.reset()

    def step(self):
        return self.cproblem.step()

    def close(self):
        self.cproblem.close()

    def render(self):
        self.cproblem.render()

    def copyObs(self, np.ndarray[float, ndim=1, mode="c"] obs not None):
        self.cproblem.copyObs(&obs[0])

    def copyAct(self, np.ndarray[float, ndim=1, mode="c"] act not None):
        self.cproblem.copyAct(&act[0])

    def copyDone(self, np.ndarray[int, ndim=1, mode="c"] done not None):
        self.cproblem.copyDone(&done[0])

    def copyDobj(self, np.ndarray[double, ndim=1, mode="c"] dob not None):
        self.cproblem.copyDobj(&dob[0])

    def isDone(self):
        return self.cproblem.isDone()

    # Attribute access
    @property
    def ninputs(self):
        return self.cproblem.ninputs
    @ninputs.setter
    def ninputs(self, ninputs):
        self.cproblem.ninputs = ninputs
    @property
    def noutputs(self):
        return self.cproblem.noutputs
    @noutputs.setter
    def noutputs(self, noutputs):
        self.cproblem.noutputs = noutputs

    # Maximum value for action
    @property
    def high(self):
        return 1.0

    # Minimum value for action
    @property
    def low(self):
        return -1.0
