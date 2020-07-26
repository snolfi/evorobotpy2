#!/bin/bash

rm ErPredprey*.so
rm ErPredprey.cpp
rm -r build
python3 setupErPredprey.py build_ext --inplace
cp ErPredprey*.so ../bin
