#!/bin/bash

# specify train and validation files here
traindata="../misc/trainer_example.txt"
valdata="../misc/tester_example.txt"

# specify model name here
exp="tweet2vec"

# model save path
modelpath="model/$exp/"
mkdir -p $modelpath

# train
echo "Training..."
python char.py $traindata $valdata $modelpath

