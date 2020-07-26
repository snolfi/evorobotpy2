/*
 * This file belong to https://github.com/snolfi/evorobotpy
 * Author: Stefano Nolfi, stefano.nolfi@istc.cnr.it

 * discrim.h, include an implementation of the discrim environment

 * This file is part of the python module ErDiscrim.so that include the following files:
 * discrim.cpp, discrim.h, robot-env.cpp, robot-env.h, utilities.cpp, utilities.h, ErDiscrim.pxd, ErDiscrim.pyx and setupErDiscrim.py
 * And can be compiled with cython and installed with the commands: cd ./evorobotpy/lib; python3 setupErDiscrim.py build_ext –inplace; cp ErDiscrim*.so ../bin
 */

#ifndef PROBLEM_H
#define PROBLEM_H

#include "utilities.h"


class Problem
{

public:
	// Void constructor
	Problem();
	// Other constructor
	Problem(double length2, double masspole2, int fixed, int ntrials, int nttrials);
	// Destructor
	~Problem();
	// Set the seed
	void seed(int s);
	// Reset trial
	void reset();
	// Perform a step of the double-pole
	double step();
	// Close the environment
	void close();
	// View the environment (graphic mode)
	void render();
	// Copy the observations
	void copyObs(float* observation);
	// Copy the action
	void copyAct(float* action);
	// Copy the termination flag
	void copyDone(int* done);
	// Copy the pointer to the vector of objects to be displayed
	void copyDobj(double* objs);
	// Check whether the episode terminated
	double isDone();

	// number of inputs
	int ninputs;
	// number of outputs
	int noutputs;

private:
	// create the environment
    void initEnvironment();
	void getObs();
	// Random generator
	RandomGenerator* rng;

};

#endif

