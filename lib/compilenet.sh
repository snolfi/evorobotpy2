#!/bin/bash

rm net*.so
rm net.cpp
rm -r build
python3 setupevonet.py build_ext --inplace
cp net*.so ../bin
