#!/bin/bash

# specify training and validation files here
traindata="data/train"
valdata="data/10K_combined_val"

# specify model name here
exp="baseline"

# model save path
modelpath="model/$exp/"
mkdir -p $modelpath

# train
echo "Training..."
python word.py $traindata $valdata $modelpath
