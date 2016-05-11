#!/bin/bash

# specify train and validation files here
traindata="data/train"
valdata="data/10K_combined_val"

# specify model name here
exp="tweet2vec"

# model save path
modelpath="model/$exp/"
mkdir -p $modelpath

# train
echo "Training..."
python char.py $traindata $valdata $modelpath

