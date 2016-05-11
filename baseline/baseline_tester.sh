#!/bin/bash

# specify test file here
fulltestdata="data/50K_combined_test"

# specify model path here
modelpath="model/baseline/"

# specify result path here
resultpath="result/baseline/"

mkdir -p $resultpath

# test
python test_word.py $fulltestdata $modelpath $resultpath
