#!/bin/bash

# specify training and validation files here
traindata="../misc/trainer_example.txt"
valdata="../misc/tester_example.txt"

# specify model name here
exp="baseline"

# model save path
modelpath="model/$exp/"
mkdir -p $modelpath

# train
echo "Training..."
python word.py $traindata $valdata $modelpath
