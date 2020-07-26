/*
 * This file belong to https://github.com/snolfi/evorobotpy
 * Author: Stefano Nolfi, stefano.nolfi@istc.cnr.it

 * predprey.h, include an implementation of the ErPredprey environment

 * This file is part of the python module ErPredprey.so that include the following files:
 * predprey.cpp, predprey.h, robot-env.cpp, robot-env.h, utilities.cpp, utilities.h, ErPredprey.pxd, ErPredprey.pyx and setupErPredprey.py
 * And can be compiled with cython and installed with the commands: cd ./evorobotpy/lib; python3 setupErPredprey.py build_ext –inplace; cp ErPredprey*.so ../bin
 */

#ifndef PROBLEM_H
#define PROBLEM_H

#include "utilities.h"


class Problem
{

public:
	// Void constructor
	Problem();
	// Destructor
	~Problem();
	// Set the seed
	void seed(int s);
	// Reset episode
	void reset();
	// Perform a step
	double step();
	// Close
	void close();
	// Render the robot and the environment
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
	int isDone();
    // number of inputs
    int ninputs;
    // number of outputs
    int noutputs;

private:
	// create the environment
    void initEnvironment();
    // compute the state of the observation
	void getObs();
	// Random generator
	RandomGenerator* rng;

};

#endif

