#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/snolfi/evorobotpy
   and has been written by Stefano Nolfi, stefano.nolfi@istc.cnr.it
   requires es.py, policy.py, and evoalgo.py 
"""

import numpy as np
from numpy import zeros, dot, sqrt
import math
import time
from evoalgo import EvoAlgo
from utils import descendent_sort
import os
import configparser


# Evolve with SSS
class Algo(EvoAlgo):
    def __init__(self, env, policy, seed, fileini, filedir):
        EvoAlgo.__init__(self, env, policy, seed, fileini, filedir)

    def loadhyperparameters(self):

        if os.path.isfile(self.fileini):

            config = configparser.ConfigParser()
            config.read(self.fileini)
            self.maxsteps = 1000000
            self.popsize = 20
            self.mutation = 0.02
            self.saveeach = 60
            options = config.options("ALGO")
            for o in options:
                found = 0
                if o == "maxmsteps":
                    self.maxsteps = config.getint("ALGO","maxmsteps") * 1000000
                    found = 1
                if o == "mutation":
                    self.mutation = config.getfloat("ALGO","mutation")
                    found = 1
                if o == "popsize":
                    self.popsize = config.getint("ALGO","popsize")
                    found = 1
                if o == "saveeach":
                    self.saveeach = config.getint("ALGO","saveeach")
                    found = 1

                if found == 0:
                    print("\033[1mOption %s in section [ALGO] of %s file is unknown\033[0m" % (o, filename))
                    print("available hyperparameters are: ")
                    print("maxmsteps [integer]       : max number of (million) steps (default 1)")
                    print("popsize [int]             : popsize (20)")
                    print("mutation [float]          : mutation (default 0.02)")
                    print("saveeach [integer]        : save file every N minutes (default 60)")

                    sys.exit()
        else:
            print("\033[1mERROR: configuration file %s does not exist\033[0m" % (self.fileini))

    def save(self, ceval, cgen, bfit, bgfit, avefit, aveweights):
            self.save()            #  save the best agent, the best postevaluated agent, and progress data across generations
            fname = self.filedir + "/S" + str(self.seed) + ".fit"  
            fp = open(fname, "w")  # save summary
            fp.write('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f cbestfit %.2f cbestgfit %.2f avgfit %.2f weightsize %.2f \n' %
                      (self.seed, ceval / float(self.maxsteps) * 100, cgen, ceval / 1000000, self.bestfit, self.bestgfit, bfit, bgfit, avefit, aveweights))
            fp.close()

    def run(self):

        self.loadhyperparameters()                 # initialize hyperparameters
        start_time = time.time()                   # start time
        nparams = self.policy.nparams              # number of parameters
        ceval = 0                                  # current evaluation
        cgen = 0                                   # current generation
        rg = np.random.RandomState(self.seed)      # create a random generator and initialize the seed
        pop = rg.randn(self.popsize, nparams)      # population
        fitness = zeros(self.popsize)              # fitness
        self.stat = np.arange(0, dtype=np.float64) # initialize vector containing performance across generations

        assert ((self.popsize % 2) == 0), print("the size of the population should be odd")

        # initialze the population
        for i in range(self.popsize):
            pop[i] = self.policy.get_trainable_flat()       

        print("SSS: seed %d maxmsteps %d popSize %d noiseStdDev %lf nparams %d" % (self.seed, self.maxsteps / 1000000, self.popsize, self.mutation, nparams))

        # main loop
        elapsed = 0
        while (ceval < self.maxsteps):
            
            cgen += 1

            # If normalize=1 we update the normalization vectors
            if (self.policy.normalize == 1):
                self.policy.nn.updateNormalizationVectors()
                
            self.env.seed(self.policy.get_seed + cgen)        # set the environment seed, it changes every generation
            self.policy.nn.seed(self.policy.get_seed + cgen)  # set the policy seed, it changes every generation
            
            # Evaluate the population
            for i in range(self.popsize):                           
                self.policy.set_trainable_flat(pop[i])        # set policy parameters
                eval_rews, eval_length = self.policy.rollout(self.policy.ntrials)  # evaluate the individual
                fitness[i] = eval_rews                        # store fitness
                ceval += eval_length                          # Update the number of evaluations
                self.updateBest(fitness[i], pop[i])           # Update data if the current offspring is better than current best

            fitness, index = descendent_sort(fitness)         # create an index with the ID of the individuals sorted for fitness
            bfit = fitness[index[0]]
            self.updateBest(bfit, pop[index[0]])              # eventually update the genotype/fitness of the best individual so far

            # Postevaluate the best individual
            self.env.seed(self.policy.get_seed + 100000)      # set the environmental seed, always the same for the same seed
            self.policy.nn.seed(self.policy.get_seed + 100000)# set the policy seed, always the same for the same seed
            self.policy.set_trainable_flat(pop[index[0]])     # set the parameters of the policy
            eval_rews, eval_length = self.policy.rollout(self.policy.ntrials)
            bgfit = eval_rews
            ceval += eval_length
            self.updateBestg(bgfit, pop[index[0]])            # eventually update the genotype/fitness of the best post-evaluated individual

            # replace the worst half of the population with a mutated copy of the first half of the population
            halfpopsize = int(self.popsize/2)
            for i in range(halfpopsize):
                pop[index[i+halfpopsize]] = pop[index[i]] + (rg.randn(1, nparams) * self.mutation)              

            # display info
            print('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f cbestfit %.2f cbestgfit %.2f avgfit %.2f weightsize %.2f' %
                      (self.seed, ceval / float(self.maxsteps) * 100, cgen, ceval / 1000000, self.bestfit, self.bestgfit, bfit, bgfit, np.average(fitness), np.average(np.absolute(pop[index[0]]))))

            # store data throughout generations
            self.stat = np.append(self.stat, [ceval, self.bestfit, self.bestgfit, bfit, bgfit, np.average(fitness)])

            # save data
            if ((time.time() - self.last_save_time) > (self.saveeach * 60)):
                self.save(ceval, cgen, bfit, bgfit, np.average(fitness), np.average(np.absolute(pop[index[0]])))
                self.last_save_time = time.time()  

        self.save(ceval, cgen, bfit, bgfit, np.average(fitness), np.average(np.absolute(pop[index[0]])))
        end_time = time.time()
        print('Simulation time: %dm%ds ' % (divmod(end_time - start_time, 60)))
