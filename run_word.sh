#!/bin/bash

traindata="data/acl/medium_500_19K/train"
valdata="data/acl/medium_500_19K/10K_combined_val"
fulltestdata="data/acl/medium_500_19K/50K_combined_test"
raretestdata="data/acl/medium_500_19K/2K_rare_test"
freqtestdata="data/acl/medium_500_19K/2K_freq_test"

exp="final_word"
modelpath="model/$exp/"
resultpath="result/$exp/"

mkdir -p $modelpath
mkdir -p $resultpath
mkdir logs

# train
echo "Training..."
python word.py $traindata $valdata $modelpath > logs/${exp}.log

# test
echo "Testing..."
python test_word.py $valdata $modelpath $resultpath > logs/${exp}_val.log
echo -e "\nResults on validation set:"
tail -3 logs/${exp}_val.log

python test_word.py $fulltestdata $modelpath $resultpath > logs/${exp}_fulltest.log
echo -e "\nResults on full test set:"
tail -3 logs/${exp}_fulltest.log

python test_word.py $raretestdata $modelpath $resultpath > logs/${exp}_raretest.log
echo -e "\nResults on rare words test set:"
tail -3 logs/${exp}_raretest.log

python test_word.py $freqtestdata $modelpath $resultpath > logs/${exp}_freqtest.log
echo -e "\nResults on frequent words test set:"
tail -3 logs/${exp}_freqtest.log
