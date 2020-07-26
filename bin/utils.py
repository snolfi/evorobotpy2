#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/snolfi/evorobotpy
   and has been written by Stefano Nolfi and Paolo Pagliuca, stefano.nolfi@istc.cnr.it, paolo.pagliuca@istc.cnr.it

   utils.py include sorting functions

"""
import numpy as np

# Sorting functions

# Descendent sorting
def descendent_sort(vect):
    # Copy of the vector
    tmpv = np.copy(vect)
    n = len(tmpv)
    # Index list
    index = np.arange(n, dtype=np.int32)
    i = 0
    while i < n:
        # Look for maximum
        maxv = tmpv[0]
        maxi = 0
        j = 1
        while j < n:
            if tmpv[j] > maxv:
                maxv = tmpv[j]
                maxi = j
            j += 1
        vect[i] = tmpv[maxi]
        index[i] = maxi
        i += 1
        # Set invalid value
        tmpv[maxi] = -999999999999.0
    return vect, index

# Ascendent sorting
def ascendent_sort(vect):
    # Copy of the vector
    tmpv = np.copy(vect)
    n = len(tmpv)
    # Index list
    index = np.arange(n, dtype=np.int32)
    i = 0
    while i < n:
        # Look for maximum
        minv = tmpv[0]
        mini = 0
        j = 1
        while j < n:
            if tmpv[j] < minv:
                minv = tmpv[j]
                mini = j
            j += 1
        vect[i] = tmpv[mini]
        index[i] = mini
        i += 1
        # Set invalid value
        tmpv[mini] = 999999999999.0
    return vect, index

