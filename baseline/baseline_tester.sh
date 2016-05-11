#!/bin/bash

# specify test file here
fulltestdata="../misc/tester_example.txt"

# specify model path here
modelpath="model/baseline/"

# specify result path here
resultpath="result/"

mkdir -p $resultpath

# test
python test_word.py $fulltestdata $modelpath $resultpath
