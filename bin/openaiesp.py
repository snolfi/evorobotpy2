#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/snolfi/evorobotpy
   and has been written by Stefano Nolfi and Paolo Pagliuca, stefano.nolfi@istc.cnr.it, paolo.pagliuca@istc.cnr.it
   salimans.py include an implementation of the OpenAI-ES algorithm described in
   Salimans T., Ho J., Chen X., Sidor S & Sutskever I. (2017). Evolution strategies as a scalable alternative to reinforcement learning. arXiv:1703.03864v2
   requires es.py, policy.py, and evoalgo.py 
"""

import numpy as np
from numpy import zeros, ones, dot, sqrt
import math
import time
from mpi4py import MPI
from evoalgo import EvoAlgo
from utils import ascendent_sort
import sys
import os
import configparser

# Parallel implementation of Open-AI-ES algorithm developed by Salimans et al. (2017)
# the workers evaluate a fraction of the population in parallel
# the master post-evaluate the best sample of the last generation and eventually update the input normalization vector

class Algo(EvoAlgo):
    def __init__(self, env, policy, seed, fileini, filedir):
        EvoAlgo.__init__(self, env, policy, seed, fileini, filedir)

    def loadhyperparameters(self):

        if os.path.isfile(self.fileini):

            config = configparser.ConfigParser()
            config.read(self.fileini)
            self.maxsteps = 1000000
            self.stepsize = 0.01
            self.batchSize = 20
            self.noiseStdDev = 0.02
            self.wdecay = 0
            self.symseed = 1
            self.saveeach = 60
            options = config.options("ALGO")
            for o in options:
                found = 0
                if o == "maxmsteps":
                    self.maxsteps = config.getint("ALGO","maxmsteps") * 1000000
                    found = 1
                if o == "stepsize":
                    self.stepsize = config.getfloat("ALGO","stepsize")
                    found = 1
                if o == "noisestddev":
                    self.noiseStdDev = config.getfloat("ALGO","noiseStdDev")
                    found = 1
                if o == "samplesize":
                    self.batchSize = config.getint("ALGO","sampleSize")
                    found = 1
                if o == "wdecay":
                    self.wdecay = config.getint("ALGO","wdecay")
                    found = 1
                if o == "symseed":
                    self.symseed = config.getint("ALGO","symseed")
                    found = 1
                if o == "saveeach":
                    self.saveeach = config.getint("ALGO","saveeach")
                    found = 1

                if found == 0:
                    print("\033[1mOption %s in section [ALGO] of %s file is unknown\033[0m" % (o, self.fileini))
                    print("available hyperparameters are: ")
                    print("maxmsteps [integer]       : max number of (million) steps (default 1)")
                    print("stepsize [float]          : learning stepsize (default 0.01)")
                    print("samplesize [int]          : popsize/2 (default 20)")
                    print("noiseStdDev [float]       : samples noise (default 0.02)")
                    print("wdecay [0/2]              : weight decay (default 0), 1 = L1, 2 = L2")
                    print("symseed [0/1]             : same environmental seed to evaluate symmetrical samples [default 1]")
                    print("saveeach [integer]        : save file every N minutes (default 60)")

                    sys.exit()
        else:
            print("\033[1mERROR: configuration file %s does not exist\033[0m" % (self.fileini))
    


    def setProcess(self, n_workers, comm, rank):
        self.loadhyperparameters()               # load parameters
        self.n_workers = n_workers               # number of workers, includes the master
        self.rank = rank                         # worker id
        self.comm = comm                         # 
        self.center = np.copy(self.policy.get_trainable_flat())  # the initial centroid
        self.nparams = len(self.center)          # number of adaptive parameters
        self.n_worker_samples = int(self.batchSize / (self.n_workers - 1)) # number of sample evaluated by each worker
        self.id = (self.rank - 1)                # id of the process (master has id -1)
        self.cgen = 0                            # currrent generation
        self.fitness = ones(self.n_workers * (self.n_worker_samples * 2)) # vector of fitness filled by the master and the workers
        self.evals = zeros(self.n_workers, dtype=np.int32)  #vector of evaluation steps filled by the master and by the workers
        self.samplefitness = zeros(self.batchSize * 2) # the fitness of the samples
        self.samples = None                      # the random samples
        self.m = zeros(self.nparams)             # Adam: momentum vector 
        self.v = zeros(self.nparams)             # Adam: second momentum vector (adam)
        self.epsilon = 1e-08                     # Adam: To avoid numerical issues with division by zero...
        self.beta1 = 0.9                         # Adam: beta1
        self.beta2 = 0.999                       # Adam: beta2
        self.bestgfit = -99999999                # the best generalization fitness
        self.bfit = 0                            # the fitness of the best sample
        self.gfit = 0                            # the postevaluation fitness of the best sample of last generation
        self.rs = None                           # random number generator
        if self.policy.normalize == 1:           # normalization vector
            self.normvector = np.arange(self.n_workers * (self.policy.ninputs * 2), dtype=np.float64)  # normalization vector broadcasted to workers
        self.inormepisodes = self.batchSize * 2 * self.policy.ntrials / 100.0 # number of normalization episode for generation (1% of generation episodes)
        self.tnormepisodes = 0.0                 # total epsidoes in which normalization data should be collected so far
        self.normepisodes = 0                    # numer of episodes in which normalization data has been actually collected so far

    def savedata(self):
        # save best postevaluated so far
        fname = self.filedir + "/bestgS" + str(self.seed)
        np.save(fname, self.bestgsol)
        # save best so far
        fname = self.filedir + "/bestS" + str(self.seed)
        np.save(fname, self.bestsol)
        # save statistics
        fname = self.filedir + "/statS" + str(self.seed)
        np.save(fname, self.stat)
        # save summary statistics
        fname = self.filedir + "/S" + str(self.seed) + ".fit"
        fp = open(fname, "w")
        fp.write('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f bestsam %.2f avgfit %.2f paramsize %.2f \n' %
             (self.seed, self.steps / float(self.maxsteps) * 100, self.cgen, self.steps / 1000000, self.bestfit, self.bestgfit, self.bfit, self.avgfit, self.avecenter))
        fp.close()

        
    def evaluate(self):
        global none_val
        seed_worker = self.seed + self.cgen * self.batchSize  # Set the seed for current generation (master and workers have the same seed)
        self.rs = np.random.RandomState(seed_worker)
        self.samples = self.rs.randn(self.batchSize, self.nparams)
        self.cgen += 1
        fitness_worker = ones(self.n_worker_samples * 2)
        ceval = 0
        
        if self.rank == 0:                                      # Master: postevaluate the best sample of last generation
            gfit = 0
            if self.bestsol is not None:
                self.policy.set_trainable_flat(self.bestsol)
                self.tnormepisodes += self.inormepisodes
                normalizationdatacollected = False
                for t in range(self.policy.nttrials):
                    if self.policy.normalize == 1 and self.normepisodes < self.tnormepisodes:
                        self.policy.nn.normphase(1)
                        self.normepisodes += 1
                        normalizationdatacollected = True
                    else:
                        self.policy.nn.normphase(0)
                    eval_rews, eval_length = self.policy.rollout(1, seed=(self.seed + 100000 + t))
                    gfit += eval_rews               
                    ceval += eval_length
                self.updateBestg(gfit / self.policy.nttrials, self.bestsol)
                if normalizationdatacollected:
                    self.policy.nn.updateNormalizationVectors()  # update the normalization vectors with the new data collected
            else:
                self.policy.nn.getNormalizationVectors()         # update the normalization vector accessible in python with that initialized by evonet
        else:      
            candidate = np.arange(self.nparams, dtype=np.float64)
            for b in range(self.n_worker_samples):               # Worker (evaluate a fraction of the population)
                for bb in range(2):
                    if (bb == 0):
                        candidate = self.center + self.samples[(self.id * self.n_worker_samples) + b,:] * self.noiseStdDev
                    else:
                        candidate = self.center - self.samples[(self.id * self.n_worker_samples) + b,:] * self.noiseStdDev
                    self.policy.set_trainable_flat(candidate)
                    self.policy.nn.normphase(0)   # workers never collect normalization data
                    eval_rews, eval_length = self.policy.rollout(self.policy.ntrials, seed=(self.seed + (self.cgen * self.batchSize) + (self.id * self.n_worker_samples) + b))
                    fitness_worker[b*2+bb] = eval_rews
                    ceval += eval_length
        ceval = np.asarray([ceval], dtype=np.int32)
        return fitness_worker, ceval


    def optimize(self):

        fitness, index = ascendent_sort(self.samplefitness)       # sort the fitness
        self.avgfit = np.average(fitness)                         # compute the average fitness                   

        self.bfit = fitness[(self.batchSize * 2) - 1]
        bidx = index[(self.batchSize * 2) - 1]  
        if ((bidx % 2) == 0):                                     # regenerate the genotype of the best samples
            bestid = int(bidx / 2)
            self.bestsol = self.center + self.samples[bestid] * self.noiseStdDev  
        else:
            bestid = int(bidx / 2)
            self.bestsol = self.center - self.samples[bestid] * self.noiseStdDev

        if self.rank == 0:
            self.updateBest(self.bfit, self.bestsol)              # Stored if it is the best obtained so far 
            
        popsize = self.batchSize * 2                              # compute a vector of utilities [-0.5,0.5]
        utilities = zeros(popsize)
        for i in range(popsize):
            utilities[index[i]] = i
        utilities /= (popsize - 1)
        utilities -= 0.5
        
        weights = zeros(self.batchSize)                           # Assign the weights (utility) to samples on the basis of their fitness rank
        for i in range(self.batchSize):
            idx = 2 * i
            weights[i] = (utilities[idx] - utilities[idx + 1])    # merge the utility of symmetric samples

        g = 0.0
        i = 0
        while i < self.batchSize:                                 # Compute the gradient (the dot product of the samples for their utilities)
            gsize = -1
            if self.batchSize - i < 500:                          # if the popsize is larger than 500, compute the gradient for multiple sub-populations
                gsize = self.batchSize - i
            else:
                gsize = 500
            g += dot(weights[i:i + gsize], self.samples[i:i + gsize,:]) 
            i += gsize
        g /= popsize                                              # normalize the gradient for the popsize
        
        if self.wdecay == 1:
            globalg = -g + 0.005 * self.center                    # apply weight decay
        else:
            globalg = -g

        # adam stochastic optimizer
        a = self.stepsize * sqrt(1.0 - self.beta2 ** self.cgen) / (1.0 - self.beta1 ** self.cgen)
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (globalg * globalg)
        dCenter = -a * self.m / (sqrt(self.v) + self.epsilon)
        
        self.center += dCenter                                    # move the center in the direction of the momentum vectors
        self.avecenter = np.average(np.absolute(self.center))      

    def update_normvector(self):
        if self.rank > 0:                                                # workers overwrite their normalization vector with the vector received from the master
            for i in range(self.policy.ninputs * 2):
                self.policy.normvector[i] = np.copy(self.normvector[i])  
            self.policy.nn.setNormalizationVectors()

    def run(self):

        start_time = time.time()
        last_save_time = start_time
        elapsed = 0
        self.steps = 0
        if self.rank ==0:
            print("Salimans: seed %d maxmsteps %d batchSize %d stepsize %lf noiseStdDev %lf wdecay %d symseed %d nparams %d" % (self.seed, self.maxsteps / 1000000, self.batchSize, self.stepsize, self.noiseStdDev, self.wdecay, self.symseed, self.nparams))

        while (self.steps < self.maxsteps):

            
            fitness_worker, weval = self.evaluate()   # evaluate sample (each worker evaluate a fraction of the population) 
            self.comm.Allgatherv(fitness_worker, [self.fitness, MPI.DOUBLE])                # brodcast fitness value to all workers
            self.comm.Allgatherv(weval, [self.evals, MPI.INT])                              # broadcast number of steps performed to all workers
            if self.policy.normalize == 1:
                self.comm.Allgatherv(self.policy.normvector, [self.normvector, MPI.DOUBLE]) # broadcast normalization vector (it is update from the master, the vectors of the workers are ignored)              

            self.samplefitness = self.fitness[(self.n_worker_samples * 2):] # Merge the fitness of the workers by discarding the vector returned by the master
            self.steps += np.sum(self.evals)          # Update the total number of steps performed so far
            
            self.optimize()                           # estimate the gradient and move the centroid in the gradient direction

            self.stat = np.append(self.stat, [self.steps, self.bestfit, self.bestgfit, self.bfit, self.avgfit, self.avecenter])  # store performance across generations
            if self.rank == 0 and ((time.time() - last_save_time) > (self.saveeach * 60)):
                self.savedata()                       # save data on files
                last_save_time = time.time()

            if self.policy.normalize == 1: 
                self.update_normvector()              # the workers overwrite their normalization vector with the vector received from the master

            if self.rank == 0:
                print('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f bestsam %.2f avg %.2f weightsize %.2f' %
                      (self.seed, self.steps / float(self.maxsteps) * 100, self.cgen, self.steps / 1000000, self.bestfit, self.bestgfit, self.bfit, self.avgfit, self.avecenter))

        if self.rank == 0:
            self.savedata()                           # save data at the end of evolution

        # print simulation time
        end_time = time.time()
        print('Simulation time: %dm%ds ' % (divmod(end_time - start_time, 60)))

