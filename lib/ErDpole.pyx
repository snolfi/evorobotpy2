"""
This file belong to https://github.com/snolfi/evorobotpy
Author: Stefano Nolfi, stefano.nolfi@istc.cnr.it

ErDpole.pyx, python wrapper for dpole.cpp

This file is part of the python module ErDpole.so that include the following files:
dpole.cpp, dpole.h, utilities.cpp, utilities.h, ErDpole.pxd, ErDpole.pyx and setupErDpole.py
And can be compiled with cython and installed with the commands: cd ./evorobotpy/lib; python3 setupErDpole.py build_ext –inplace; cp ErDpole*.so ../bin
"""

# distutils: language=c++

import cython
# import both numpy and the Cython declarations for numpy
import numpy as np
import time
cimport numpy as np
from libcpp cimport bool
from ErDpole cimport Problem

# PyErProblem
cdef class PyErProblem:
    cdef Problem cproblem

    def __cinit__(self):
        self.cproblem = Problem()

    def seed(self, s):
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

    # Attribute access
    @property
    def ninputs(self):
        return self.cproblem.m_ninputs
    @ninputs.setter
    def ninputs(self, ninputs):
        self.cproblem.m_ninputs = ninputs
    @property
    def noutputs(self):
        return self.cproblem.m_noutputs
    @noutputs.setter
    def noutputs(self, noutputs):
        self.cproblem.m_noutputs = noutputs

    # Maximum value for action
    @property
    def high(self):
        return self.cproblem.m_high

    # Minimum value for action
    @property
    def low(self):
        return self.cproblem.m_low
