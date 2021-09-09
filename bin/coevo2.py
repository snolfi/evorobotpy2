#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/snolfi/evorobotpy
   and has been written by Stefano Nolfi, stefano.nolfi@istc.cnr.it

   coevo2.py include an implementation of an competitive co-evolutionary algorithm analogous
   to that described in:
   Simione L and Nolfi S. (2019). Long-Term Progress and Behavior Complexification in Competitive Co-Evolution, arXiv:1909.08303.

   Requires es.py policy.py and evoalgo.py
   Also requires the net.so library that can be obtained by compiling with cython the following files contained in the ./lib directory:
   evonet.cpp, evonet.h, utilities.cpp, utilities.h, net.pxd, net.pyx and setupevonet.py   
   with the commands: cd ./evorobotpy/lib; python3 setupevonet.py build_ext –inplace; cp net*.so ../bin 
"""

import numpy as np
from numpy import zeros, dot, sqrt
import math
import time
from evoalgo import EvoAlgo
from utils import ascendent_sort
import random
import os
import sys
import configparser

# competitive coevolutionary algorithm operating on two populations
class Algo(EvoAlgo):
    def __init__(self, env, policy, seed, fileini, filedir):
        EvoAlgo.__init__(self, env, policy, seed, fileini, filedir)

    def loadhyperparameters(self):

        if os.path.isfile(self.fileini):

            config = configparser.ConfigParser()
            config.read(self.fileini)
            self.popsize = 80
            self.selsize = 10
            self.ngenerations = 1000
            self.stepsize = 0.01
            self.batchSize = 20
            self.noiseStdDev = 0.02
            self.wdecay = 0
            self.saveeach = 100
            options = config.options("ALGO")
            for o in options:
                found = 0
                if o == "ngenerations":
                    self.ngenerations = config.getint("ALGO","ngenerations")
                    found = 1
                if o == "selsize":
                    self.selsize = config.getint("ALGO","selsize")
                    found = 1
                if o == "popsize":
                    self.popsize = config.getint("ALGO","popsize")
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
                if o == "saveeach":
                    self.saveeach = config.getint("ALGO","saveeach")
                    found = 1

                if found == 0:
                    print("\033[1mOption %s in section [ALGO] of %s file is unknown\033[0m" % (o, filename))
                    print("available hyperparameters are: ")
                    print("ngenerations [integer]    : max number of generations (default 200)")
                    print("popsize [integer]         : popsize (default 40)")
                    print("selsize [integer]         : number selected agents (default 10)")
                    print("stepsize [float]          : learning stepsize (default 0.01)")
                    print("samplesize [int]          : samplesize/2 (default 20)")
                    print("noiseStdDev [float]       : samples noise (default 0.02)")
                    print("wdecay [0/2]              : weight decay (default 0), 1 = L1, 2 = L2")
                    print("saveeach [integer]        : save file every N generations (default 100)")

                    sys.exit()
        else:
            print("\033[1mERROR: configuration file %s does not exist\033[0m" % (self.fileini))
    


    def run(self):

        self.loadhyperparameters()           # load hyperparameters

        seed = self.seed
        self.rs = np.random.RandomState(self.seed)

        # Extract the number of parameters
        nparams = int(self.policy.nparams / 2)                                   # parameters required for a single individul
        
        # allocate and vectors
        pop = []                                                                 # the populations (the individuals of the second pop follow) 
        popm = []                                                                # the momentum of the populations
        popv = []                                                                # the squared momentum of the populations
        self.candidate = np.arange(nparams, dtype=np.float64)                    # the vector containing varied parameters      
        self.fmatrix = np.zeros((self.popsize+self.selsize, self.popsize++self.selsize), dtype=np.float64) # the fitness of each individual against each competitor
                                                                                 # the additional lines and columns contain data of evolving individuals
       
        self.selp = np.arange(nparams*self.selsize, dtype=np.float64)            # the parameters of the selected individuals
        self.selm = np.arange(nparams*self.selsize, dtype=np.float64)            # the momentum of the selected individuals
        self.selv = np.arange(nparams*self.selsize, dtype=np.float64)            # the squared-momentum of the selected individuals                         
        self.selcomp = np.arange(nparams*self.selsize, dtype=np.float64)         # the parameters of the selected competitors
        self.selp = np.resize(self.selp, (self.selsize, nparams))  
        self.selm = np.resize(self.selm, (self.selsize, nparams))
        self.selv = np.resize(self.selv, (self.selsize, nparams))
        self.selcomp = np.resize(self.selcomp, (self.selsize, nparams))
        # initialize population vectors 
        for i in range(self.popsize*2):
            self.policy.nn.initWeights()
            randomparams = np.copy(self.policy.get_trainable_flat())
            pop.append(randomparams[:nparams])
            popm.append(zeros(nparams))
            popv.append(zeros(nparams))
        pop = np.asarray(pop)
        popm = np.asarray(popm)
        popv = np.asarray(popv)

        print("Coevo2 seed %d Popsize %d %d batchSize %d stepsize %lf noiseStdDev %lf wdecay %d nparams %d" % (self.seed, self.popsize, self.selsize, self.batchSize, self.stepsize, self.noiseStdDev, self.wdecay, nparams))

        # evaluate pop1 against pop2
        print("gen %d eval pop1 against pop2" % (0))
        for i1 in range(self.popsize):
            for i2 in range(self.popsize):
                self.policy.set_trainable_flat(np.concatenate((pop[i1], pop[self.popsize+i2])))
                eval_rews, eval_length = self.policy.rollout(1)
                self.fmatrix[i1][i2] = eval_rews
                #print("%.2f " % (eval_rews), end = '')
            #print("")
        filename = "S%dG0.npy" % (seed)
        np.save(filename, pop)
        filename = "S%dFitG0.npy" % (seed)
        np.save(filename, self.fmatrix)  

        # main loop
        self.evopop = 0 # whether the first or the second pop evolve
        for gen in range(self.ngenerations):
            # chooses the selected competitors
            #self.selc = random.sample(range(self.popsize), self.selsize)
            self.selc = self.seldiffcomp()
            print("gen %d competitors: " % (gen), end = '')
            print(self.selc)
            # chooses the selected individuals 
            self.seli = random.sample(range(self.popsize), self.selsize)
            # update the matrix of selected individuals (with associated momentum vectors) and the matrix of selected competitors
            for sind in range(self.selsize):
                if (self.evopop == 0):
                    for p in range(nparams):
                        self.selp[sind][p] = pop[self.seli[sind]][p]
                        self.selm[sind][p] = popm[self.seli[sind]][p]
                        self.selv[sind][p] = popv[self.seli[sind]][p]
                        self.selcomp[sind][p] = pop[self.popsize+self.selc[sind]][p]
                else:
                    for p in range(nparams):
                        self.selp[sind][p] = pop[self.popsize+self.seli[sind]][p]
                        self.selm[sind][p] = popm[self.popsize+self.seli[sind]][p]
                        self.selv[sind][p] = popv[self.popsize+self.seli[sind]][p]
                        self.selcomp[sind][p] = pop[self.selc[sind]][p]
            # evolve individuals
            for sind in range(self.selsize):
                self.runphase(sind, nparams)
            # test evolving individual agaist all competitors
            print("gen %d postevaluate against all competitors" % (gen))
            for i1 in range(self.selsize):
                for i2 in range(self.popsize):
                    if (self.evopop == 0):
                        self.policy.set_trainable_flat(np.concatenate((self.selp[i1], pop[self.popsize+i2])))
                    else:
                        self.policy.set_trainable_flat(np.concatenate((pop[i2], self.selp[i1])))
                    eval_rews, eval_length = self.policy.rollout(1)
                    if (self.evopop == 0):
                        self.fmatrix[self.popsize+i1][i2] = eval_rews # additional rows of the popsize*popsize matrix
                    else:
                        self.fmatrix[i2][self.popsize+i1] = eval_rews # additional columns of the popsize*popsize matrix                      
            if (self.evopop == 0):
                # average lines (pop 1), submatrix of popsize+selsize raws and popsize columns
                fm = self.fmatrix[0:self.popsize+self.selsize,0:self.popsize].mean(axis=1, dtype='float')  
                orderfm  = fm.argsort()    # sort ascending order
            else:
                # average columns (pop 2), submatrix of popsize raws and popsize+selsize columns
                fm = self.fmatrix[0:self.popsize,0:self.popsize+self.selsize].mean(axis=0, dtype='float')  
                fm = 1.0 - fm              # transfor fitness from pop1 to pop2 point of view
                orderfm  = fm.argsort()    # sort ascending order
                
            # replace the worst population individuals with the evolving individuals that ootperform them in postevaluation
            replaced = 0
            while (orderfm[replaced] >= self.popsize and replaced < (self.popsize - 1)):
                replaced  += 1
            localprog = 0
            for i in range(self.popsize):
                if (orderfm[i+self.selsize] >= self.popsize): # evolving individual ranked among the best
                    evoi = orderfm[i+self.selsize] - self.popsize
                    worsei = orderfm[replaced]
                    print("%d->%d %.2f " % (evoi,worsei, fm[self.popsize + evoi] - fm[worsei]), end ='')
                    if (self.evopop == 0):
                        for p in range(nparams):
                            pop[worsei][p]  = self.selp[evoi][p] 
                            popm[worsei][p] = self.selm[evoi][p]
                            popv[worsei][p] = self.selv[evoi][p]
                        localprog += fm[self.popsize + evoi] - fm[worsei]
                        for c in range(self.popsize):
                            self.fmatrix[worsei][c] = self.fmatrix[evoi+self.popsize][c]
                    else:
                        for p in range(nparams):
                            pop[worsei+self.popsize][p]  = self.selp[evoi][p] 
                            popm[worsei+self.popsize][p] = self.selm[evoi][p]
                            popv[worsei+self.popsize][p] = self.selv[evoi][p]
                        localprog += fm[self.popsize + evoi] - fm[worsei]
                        for c in range(self.popsize):
                            self.fmatrix[c][worsei] = self.fmatrix[c][evoi+self.popsize]                        
                    replaced += 1
                    while (orderfm[replaced] >= self.popsize and replaced < (self.popsize - 1)):
                        replaced  += 1
            print("local progress %.2f " % (localprog / self.selsize))
            # save evolving populations and fitness matrix
            if (((gen + 1) % self.saveeach) == 0):
                filename = "S%dG%d.npy" % (seed, gen + 1)
                np.save(filename, pop)
                filename = "S%dFitG%d.npy" % (seed, gen + 1)
                np.save(filename, self.fmatrix)
            if (((gen + 1) % (self.saveeach * 10)) == 0):
                filename = "S%dG%dm.npy" % (seed, gen + 1)
                np.save(filename, popm)
                filename = "S%dG%dv.npy" % (seed, gen + 1)
                np.save(filename, popv)
            fm = self.fmatrix[0:self.popsize,0:self.popsize].mean(dtype='float')
            print("seed %d gen %d popfit %.2f %.2f weights %.2f" % (seed, gen, fm, 1.0 - fm, np.average(np.absolute(pop))))
            # changes the evolving population 
            self.evopop += 1
            if (self.evopop > 1):
                self.evopop = 0

    # select differentiated competitors
    # the first is chosen randomly, the next are those that achieved the maximum different performance 
    def seldiffcomp(self):
        comp = np.zeros(self.selsize)
        unselected = np.arange(self.popsize)
        selected = []
        # first competitor is selected randomly
        selind = random.randint(0, self.popsize-1)
        selected.append(selind)
        unselected = np.delete(unselected, selind)
        # select the competitor that differ more with respect to already selected competitors
        while (len(selected) < self.selsize):
            selind = 0
            maxdiff = 0
            for i1 in range(len(unselected)):
                diff = 0
                for i2 in range(len(selected)):
                    for i3 in range(self.popsize):
                        if (self.evopop == 0):
                            diff += abs(self.fmatrix[i3][unselected[i1]] - self.fmatrix[i3][selected[i2]])
                        else:
                            diff += abs(self.fmatrix[unselected[i1]][i3] - self.fmatrix[selected[i2]][i3])
                if (diff > maxdiff):
                    selind = i1
                    maxdiff = diff
            selected.append(unselected[selind])
            unselected = np.delete(unselected, selind)
    
        return(selected)                
            
    # evolve selected individuls against selected competitors     
    def runphase(self, sind, nparams):
        
        epsilon = 1e-08 
        beta1 = 0.9
        beta2 = 0.999
        weights = zeros(self.batchSize)

        for it in range (20):
            ave_rews = 0
            # evaluate the centroid
            for i in range(self.selsize):
                if (self.evopop == 0):
                    self.policy.set_trainable_flat(np.concatenate((self.selp[sind], self.selcomp[i])))
                    eval_rews, eval_length = self.policy.rollout(1)
                    # sanity check
                    if (it == 0 and eval_rews != self.fmatrix[self.seli[sind],self.selc[i]]):
                        print("warning: sanity check failed")
                    ave_rews += eval_rews
                else:
                    self.policy.set_trainable_flat(np.concatenate((self.selcomp[i], self.selp[sind])))
                    eval_rews, eval_length = self.policy.rollout(1)
                    # sanity check
                    if (it == 0 and eval_rews != self.fmatrix[self.selc[i],self.seli[sind]]):
                        print("warning: sanity check failed")
                    ave_rews += (1.0  - eval_rews)
            ave_rews /= float(self.selsize)
            #print("centroid ", end ='')
            #for g in range(10):
                #print("%.4f " % (self.selp[sind][g+20]), end='')
            #print("");
            if (it == 0):
                print("evopop %d ind %2d : " % (self.evopop, self.seli[sind]), end = '')
            print("%.2f " % (ave_rews), end='')

            # Extract half samples from Gaussian distribution with mean 0.0 and standard deviation 1.0
            samples = self.rs.randn(self.batchSize, nparams)
            fitness = zeros(self.batchSize * 2)
            # Evaluate offspring
            for b in range(self.batchSize):
                for bb in range(2):
                    if (bb == 0):
                        for g in range(nparams):
                            self.candidate[g] = self.selp[sind][g] + samples[b,g] * self.noiseStdDev
                    else:
                        for g in range(nparams):
                            self.candidate[g] = self.selp[sind][g] - samples[b,g] * self.noiseStdDev
                    #print("candidad ", end ='')
                    #for g in range(10):
                        #print("%.4f " % (self.candidate[g+20]), end='')
                    #print("");
                    # evaluate offspring
                    ave_rews = 0
                    for c in range(self.selsize):
                        if (self.evopop == 0):
                            self.policy.set_trainable_flat(np.concatenate((self.candidate, self.selcomp[c])))
                            eval_rews, eval_length = self.policy.rollout(1)
                            ave_rews += eval_rews
                        else:
                            self.policy.set_trainable_flat(np.concatenate((self.selcomp[c], self.candidate)))
                            eval_rews, eval_length = self.policy.rollout(1)
                            ave_rews += (1.0 - eval_rews)
                        #print("f %.2f" % eval_rews)
                    fitness[b*2+bb] = ave_rews / float(self.selsize)
                    #print("%.2f " % (ave_rews / float(self.selsize)), end = '')
            # Sort by fitness and compute weighted mean into center
            fitness, index = ascendent_sort(fitness)
            # Now me must compute the symmetric weights in the range [-0.5,0.5]
            utilities = zeros(self.batchSize * 2)
            for i in range(self.batchSize * 2):
                utilities[index[i]] = i
            utilities /= (self.batchSize * 2 - 1)
            utilities -= 0.5
            # Now we assign the weights to the samples
            for i in range(self.batchSize):
                idx = 2 * i
                weights[i] = (utilities[idx] - utilities[idx + 1]) # pos - neg

            # Compute the gradient
            g = 0.0
            i = 0
            while i < self.batchSize:
                gsize = -1
                if self.batchSize - i < 500:
                    gsize = self.batchSize - i
                else:
                    gsize = 500
                g += dot(weights[i:i + gsize], samples[i:i + gsize,:]) # weights * samples
                i += gsize
            # Normalization over the number of samples
            g /= (self.batchSize * 2)
            # Weight decay
            if (self.wdecay == 1):
                globalg = -g + 0.005 * self.selp[sind]
            else:
                globalg = -g
            # ADAM stochastic optimizer
            # a = self.stepsize * sqrt(1.0 - beta2 ** cgen) / (1.0 - beta1 ** cgen)
            a = self.stepsize # bias correction is not implemented
            self.selm[sind] = beta1 * self.selm[sind] + (1.0 - beta1) * globalg
            self.selv[sind] = beta2 * self.selv[sind] + (1.0 - beta2) * (globalg * globalg)
            dCenter = -a * self.selm[sind] / (sqrt(self.selv[sind]) + epsilon)
            # update center
            self.selp[sind] += dCenter
            #for g in range(10):
                 #print("%.4f " % (self.selp[sind][g+20]), end='')
            #print("");

        # evaluate the evolving individual at the end of the evolution phase
        ave_rews = 0
        for i in range(self.selsize):
            if (self.evopop == 0):
                self.policy.set_trainable_flat(np.concatenate((self.selp[sind], self.selcomp[i])))
                eval_rews, eval_length = self.policy.rollout(1)
                ave_rews += eval_rews
            else:
                self.policy.set_trainable_flat(np.concatenate((self.selcomp[i], self.selp[sind])))
                eval_rews, eval_length = self.policy.rollout(1)
                ave_rews += (1.0  - eval_rews)
        ave_rews /= float(self.selsize)
        print("%.2f" % (ave_rews))

    def testusage(self):
        print("ERROR: To post-evaluate with the coevo algorithm you should specify with the -g parameter a string containing:")
        print("P-ng-ni (postevaluate a population) where ng is generation number and ni the number of best agents to be posteveluated")
        print("p-ng-ni (postevaluate a population without displaying the behavior)")
        print("m-ng-ngg (master tournament) where ng is the last generation number and ngg is the generation interval ")
        print("c-pop1-pop2 (population cross test) pop1 and pop2 are the name of the files containing the populations to be cross-tested ")
        sys.exit()

    def test(self, testparam):
        if testparam is None:
            self.testusage()
        if "-" not in testparam:
            self.testusage()
        seed = self.seed
        parsen = testparam.split("-")
        if (len(parsen) != 3 or not parsen[0] in ["P", "p", "m", "M","c", "C"]):
            self.testusage()
            
        # P-g-max: Test generation g (only the best max individuals)
        # P renders behavior, "p" only print fitness
        if (parsen[0] == "p" or parsen[0] == "P"):
            if (parsen[0] == "P"):
                self.policy.test = 1
                rendt = True
            else:
                self.policy.test = 0
                rendt = False
            popfile = "S%dG%d.npy" % (seed,int(parsen[1]))
            print("load %s" % (popfile))
            pop = np.load(popfile)
            popshape = pop.shape
            popsize = int(popshape[0] / 2)
            if (len(parsen) >= 3):
                maxi = int(parsen[2])
            else:
                maxi = popsize
            fmatrixfile = "S%dFitG%d.npy" % (seed, int(parsen[1]))
            fmatrix = np.load(fmatrixfile)
            fit1 =  fmatrix[0:popsize,0:popsize].mean(axis=1, dtype='float')
            rank1 = fit1.argsort()    # sort ascending order
            fit2  = fmatrix[0:popsize,0:popsize].mean(axis=0, dtype='float')  
            rank2 = fit2.argsort()     # sort ascending order
            # print the matrix loaded from file
            print("    ", end = '')
            for i1 in range(popsize):
                print("\033[1m%4d \033[0m" % (i1), end = '')
            print("")
            for i1 in range(popsize):
                print("\033[1m%3d \033[0m" % (i1), end = '')
                for i2 in range(popsize):
                    print("%.2f " % (fmatrix[i1,i2]), end = '')
                print("\033[1m%.2f\033[0m" % (fit1[i1]))
            print("    ", end = '')
            for i1 in range(popsize):
                print("\033[1m%.2f \033[0m" % (fit2[i1]), end = '')
            print("")
            # test in order of performance
            print("")
            if (not rendt):
                print("    ", end = '')
                for i2 in range(maxi):
                    print("\033[1m%4d \033[0m" % (rank2[i2]), end = '')
                print("")
            fitcol = np.zeros(maxi)
            i1 = popsize - 1
            ii1 = 0
            while (ii1 < maxi):
                if (not rendt):
                    print("\033[1m%3d \033[0m" % (rank1[i1]), end = '')
                tot_rew = 0
                for i2 in range(maxi):
                    if (rendt):
                        print("pred %d prey %d " % (rank1[i1], rank2[i2]), end = '')
                    self.policy.set_trainable_flat(np.concatenate((pop[rank1[i1]], pop[popsize+rank2[i2]])))
                    eval_rews, eval_length = self.policy.rollout(1)
                    tot_rew += eval_rews
                    fitcol[i2] += eval_rews
                    if (not rendt):
                        print("%.2f " % eval_rews, end = '')
                    if (fmatrix[rank1[i1]][rank2[i2]] != eval_rews):
                        print("warning [%.2f %.2f]" % (eval_rews, fmatrix[rank1[i1]][rank2[i2]]) , end = '')
                if (not rendt):
                    print("\033[1m%.2f\033[0m" % (tot_rew / float(maxi)))   
                i1 -= 1
                ii1 += 1
            if (not rendt):
                print("    ", end = '')
                for i2 in range(maxi):
                    print("\033[1m%.2f \033[0m" % (fitcol[i2] / float(maxi)), end = '')
                print("")
            

        # "m-n1-n2, Master tournament (only last gen), test pop of generation n1 against competitors of previous generations every n2 generations 
        if (parsen[0] == "m"):
            popfile = "S%dG%d.npy" % (seed,int(parsen[1]))
            pop = np.load(popfile)
            popshape = pop.shape
            popsize = int(popshape[0] / 2)
            self.policy.test = 0
            bestrew1 = ""
            bestrew2 = ""
            print("seed %d: postevaluation gen %d against contemporary and ancient competitors every %d generaions" % (seed, int(parsen[1]), int(parsen[2])))
            for pp in range(2):
                if (pp == 0):
                    print("pred: ", end ='', flush=True)
                else:
                    print("prey: ", end ='', flush=True)
                cgen = int(parsen[1])
                while (cgen >= 0):
                    pop2file = "S%dG%d.npy" % (seed,cgen)
                    pop2 = np.load(pop2file)
 
                    tot_rew = 0
                    max_ind_rew = 0
                    for i1 in range(popsize):
                        ind_rew = 0
                        for i2 in range(popsize):
                            if (pp == 0):
                                self.policy.set_trainable_flat(np.concatenate((pop[i1], pop2[popsize+i2])))
                            else:
                                self.policy.set_trainable_flat(np.concatenate((pop2[i1], pop[popsize+i2])))                                
                            rew, eval_length = self.policy.rollout(1)
                            tot_rew += rew
                            ind_rew += rew
                        ind_rew = ind_rew / popsize
                        if (ind_rew > max_ind_rew):
                            max_ind_rew = ind_rew
                    if (pp == 0):
                        print("%.2f " % (tot_rew / (popsize*popsize)), end = '', flush=True)
                    else:
                        print("%.2f " % (1.0 - (tot_rew / (popsize*popsize))), end = '', flush=True)
                    if (pp == 0):
                        bestrew1 += "%.2f " % (max_ind_rew)
                        #bestrew1.append(st)
                    else:
                        bestrew2 += "%.2f " % (1.0 - max_ind_rew)
                        #bestrew2.append(st)
                    cgen -= int(parsen[2])
                print("")
            print("pred-max: ", end = '')
            print(bestrew1)
            print("prey-max: ", end = '')
            print(bestrew2)
        # "M-n1-n2, Master tournament, test pop of all generations up to generation n1 against opponent of all generations, every n2 generations 
        if (parsen[0] == "M"):
            uptogen = int(parsen[1])
            everygen = int(parsen[2])
            ntests = int(uptogen / everygen)
            popfile = "S%dG%d.npy" % (seed,0)
            pop = np.load(popfile)
            popshape = pop.shape
            popsize = int(popshape[0] / 2)
            self.policy.test = 0
            master = np.zeros((ntests, ntests), dtype=np.float64) # matrix with the average performance of every generation against every other generation
            print("seed %d: postevaluation all generations up to %d against all competitors, every %d generations" % (seed, uptogen, everygen))
            for p in range(ntests):
                for pp in range(ntests):
                    popfile = "S%dG%d.npy" % (seed,p * everygen)
                    pop = np.load(popfile)
                    pop2file = "S%dG%d.npy" % (seed,pp * everygen)
                    pop2 = np.load(pop2file)
                    tot_rew = 0
                    max_ind_rew = 0
                    for i1 in range(popsize):
                        ind_rew = 0
                        for i2 in range(popsize):
                            if (pp == 0):
                                self.policy.set_trainable_flat(np.concatenate((pop[i1], pop2[popsize+i2])))
                            else:
                                self.policy.set_trainable_flat(np.concatenate((pop2[i1], pop[popsize+i2])))                                
                            rew, eval_length = self.policy.rollout(1)
                            tot_rew += rew
                            ind_rew += rew
                        ind_rew = ind_rew / popsize
                        if (ind_rew > max_ind_rew):
                            max_ind_rew = ind_rew
                    master[p][pp] = tot_rew / float(ntests * ntests)
            mfile = "masterS%d.npy" % (seed)
            np.save(mfile, master)            
        # "C-file1-file2, cross-experiment (pred and prey of file1 against themselves and against prey and pred of file2   
        if (parsen[0] == "c" or parsen[0] == "C"):
            print("crosstest of %s against %s " % (parsen[1], parsen[2]))
            self.policy.test = 0
            pop1 = np.load(parsen[1])
            popshape1 = pop1.shape
            popsize1 = int(popshape1[0] / 2)
            pop2 = np.load(parsen[2])
            popshape2 = pop2.shape
            popsize2 = int(popshape2[0] / 2)            
            assert popshape1[1] == popshape2[1], "the number of parameters in the two file is inconsistent"
            # 4 cases, pred1-prey1, pred1-prey2, pred2-prey1, pred2-prey2
            tot_rew = [0,0,0,0]
            for pp in range(4):
                if (pp == 0):
                    print("pred1-prey1: ", end ='', flush=True)
                    psizea = popsize1
                    psizeb = popsize1
                if (pp == 1):
                    print("pred1-prey2: ", end ='', flush=True)
                    psizea = popsize1
                    psizeb = popsize2                   
                if (pp == 2):
                    print("pred2-prey1: ", end ='', flush=True)
                    psizea = popsize2
                    psizeb = popsize1
                if (pp == 3):
                    print("pred2-prey2: ", end ='', flush=True)
                    psizea = popsize2
                    psizeb = popsize2
                for i1 in range(psizea):
                    for i2 in range(psizeb):
                        if (pp == 0):
                            self.policy.set_trainable_flat(np.concatenate((pop1[i1], pop1[popsize1+i2])))
                        if (pp == 1):
                            self.policy.set_trainable_flat(np.concatenate((pop1[i1], pop2[popsize2+i2])))
                        if (pp == 2):
                            self.policy.set_trainable_flat(np.concatenate((pop2[i1], pop1[popsize1+i2])))
                        if (pp == 3):
                            self.policy.set_trainable_flat(np.concatenate((pop2[i1], pop2[popsize2+i2])))                               
                        rew, eval_length = self.policy.rollout(1)
                        tot_rew[pp] += rew
                tot_rew[pp] /= (psizea*psizeb)
                print("%.2f " % (tot_rew[pp]), flush=True)
            print("pred diff: %.2f" % (tot_rew[1] - tot_rew[0]))
            print("prey diff: %.2f" % ((1.0 - tot_rew[2]) - (1.0 - tot_rew[3])))
            print("tot  diff: %.2f" % (tot_rew[1] - tot_rew[0] + (1.0 - tot_rew[2]) - (1.0 - tot_rew[3])))          




