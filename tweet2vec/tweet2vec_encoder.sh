#!/bin/bash

# specify data file here
datafile="data/tweets"

# specify model path here
modelpath="model/tweet2vec/"

# specify result path here
resultpath="result/tweet2vec/"

mkdir -p $resultpath

# test
python encode_char.py $datafile $modelpath $resultpath
