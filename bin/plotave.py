#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/snolfi/evorobotpy
   and has been written by Stefano Nolfi, stefano.nolfi@istc.cnr.it

   plotave.py extract performance in postevaluation episode from StatS?.fit files
   and print the average and the standard deviation
   
"""

import numpy as np
import sys
import os

# extract generalization data from *.fit files and plot average and standard deviation

found = False
averagen = 0
data = []

if len(sys.argv) == 1:
    cpath = os.getcwd()
    files = os.listdir(cpath)
    for f in files:
        if ("S" in f) and (".fit" in f):
            f = open(f)
            for l in f:
                for el in l.split():
                    if found:
                        averagen += 1
                        data.append(float(el))
                        found = False
                    if (el == 'bestgfit'):
                        found = True
print("")
if (averagen > 0):
    print("Average Generalization: %.2f +-%.2f (%d S*.fit files)" % (np.average(data), np.std(data), averagen))
    named = os.getcwd()
    named = named.split("/")
    fname = named[len(named)-1] + ".st"   
    fp = open(fname, "w")
    for d in data:
        fp.write("%f\n" % d)
    fp.close()
    
else:
    print("No data found")
    print("Compute the average and stdev of generalization performance")
    print("Extract data from S*.fit files: data should follow the bestgfit key")
    print("Save extracted data in a current-directory.st file")


