cdef extern from "utilities.cpp":
    pass

cdef extern from "cforaging.cpp":
    pass

# Declare the class with cdef
cdef extern from "cforaging.h":
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
        void copyDone(double* done)
        void copyDobj(double* objs)

