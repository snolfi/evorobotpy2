#!/bin/bash

rm ErDiscrim*.so
rm ErDiscrim.cpp
rm -r build
python3 setupErDiscrim.py build_ext --inplace
cp ErDiscrim*.so ../bin
