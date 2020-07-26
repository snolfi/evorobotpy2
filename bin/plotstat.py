#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/snolfi/evorobotpy
   and has been written by Stefano Nolfi, stefano.nolfi@istc.cnr.it

   plotstat.py the fitness across generation contatined in stat*.npy files

"""


import matplotlib.pyplot as plt
import numpy as np
import sys
import os

print("plotstat.py")
print("plot the fitness across generation contatined in stat*.npy files")
print("if called with a filename, plot the data of that file")
print("if called without parameters, plot the data of all available stat*.npy files")
print("")

# plot the data contained in the parameter file
# if called without parameters plot all available statS?.npy files

statsumn = 0
statavesum = 0
np.random.seed(1)

if len(sys.argv) == 1:
    cpath = os.getcwd()
    files = os.listdir(cpath)
    plt.title('statS*.npy')
    print("Plotting data contained in:")
    for f in files:
        if "statS" in f:
            print(f)
            stat = np.load(f)
            size = np.shape(stat)
            newsize = (int(size[0] / 6), 6)
            stat = np.resize(stat, newsize)
            stat = np.transpose(stat)            
            #if (statsumn == 0):
                #statl = len(stat[0])
                #statsum = np.zeros((6,statl))
            col = np.random.uniform(low=0.0, high=1.0, size=3)
            plt.plot(stat[0],stat[2],label=f, linewidth=1,  color=col)
            #statsum = statsum + stat
            #statavesum += 1
            statsumn = statsumn + 1
        #statsum = statsum / float(statavesum)
        #plt.plot(statsum[0],statsum[2],label='ave', linewidth=1,  color='r')
    if (statsumn == 0):
        print("\033[1mERROR: No stat*.npy file found\033[0m") 
    else:
        plt.legend()
        plt.show()

    
else:
    if len(sys.argv) == 2:
        stat = np.load(sys.argv[1])
        size = np.shape(stat)
        newsize = (int(size[0] / 6), 6)
        stat = np.resize(stat, newsize)
        stat = np.transpose(stat)
        plt.title(sys.argv[1])
        plt.plot(stat[0],stat[1],label='fit', linewidth=1,  color='r')
        plt.plot(stat[0],stat[2],label='gfit', linewidth=1,  color='b')
        plt.legend()
        plt.show()


