#!/bin/bash

rm ErDpole*.so
rm ErDpole.cpp
rm -r build
python3 setupErDpole.py build_ext --inplace
cp ErDpole*.so ../bin
