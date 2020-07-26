#!/bin/bash

rm ErStaybehind*.so
rm ErStaybehind.cpp
rm -r build
python3 setupErStaybehind.py build_ext --inplace
cp ErStaybehind*.so ../bin
