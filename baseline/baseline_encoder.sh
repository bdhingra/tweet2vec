#!/bin/bash

# specify data file here
datafile="data/tweets"

# specify model path here
modelpath="model/baseline/"

# specify result path here
resultpath="result/baseline/"

mkdir -p $resultpath

# test
python encode_word.py $datafile $modelpath $resultpath
