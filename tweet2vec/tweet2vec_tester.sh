#!/bin/bash

# specify test file here
fulltestdata="data/50K_combined_test"

# specify model path here
modelpath="model/tweet2vec/"

# specify result path here
resultpath="result/tweet2vec/"

mkdir -p $resultpath

# test
python test_char.py $fulltestdata $modelpath $resultpath
