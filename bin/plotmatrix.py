#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/snolfi/evorobotpy2
   and has been written by Stefano Nolfi, stefano.nolfi@istc.cnr.it

   plotmatrix.py plot the data contained in a *.npy matrix file as a colormap

"""


from matplotlib import pyplot as plt
import numpy as np
import sys
import os

print("plotmatrix.py")
print("plot the data contained in a *.npy matrix file as a colormap")
print("requires the name of the file or of the files as a parameter")
print("in case of multiple files, it plots the average data")
print("")


if len(sys.argv) == 2:
    matrix = np.load(sys.argv[1])
else:
    if len(sys.argv) > 2:
        cmatrix = np.load(sys.argv[1])
        matrix = np.zeros(cmatrix.shape)
        nmatrix = 0
        for f in sys.argv[1:]:
            cmatrix = np.load(f)
            matrix = matrix + cmatrix
            nmatrix = nmatrix + 1
        matrix = matrix / float(nmatrix)
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
#im = plt.imshow(matrix, cmap="copper_r")
im = plt.imshow(matrix, cmap="bwr")
plt.colorbar(im)
plt.show()
