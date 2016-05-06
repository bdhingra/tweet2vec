#!/bin/bash

traindata="data/train"
valdata="data/10K_combined_val"
fulltestdata="data/50K_combined_test"
raretestdata="data/2K_rare_test"
freqtestdata="data/2K_freq_test"

exp="final_char_noschedule"
modelpath="model/$exp/"
resultpath="result/$exp/"

mkdir -p $modelpath
mkdir -p $resultpath
mkdir logs

# train
echo "Training..."
python char.py $traindata $valdata $modelpath > logs/${exp}.log

# test
echo "Testing..."
python test_char.py $valdata $modelpath $resultpath > logs/${exp}_val.log
echo -e "\nResults on validation set:"
tail -3 logs/${exp}_val.log

python test_char.py $fulltestdata $modelpath $resultpath > logs/${exp}_fulltest.log
echo -e "\nResults on full test set:"
tail -3 logs/${exp}_fulltest.log

python test_char.py $raretestdata $modelpath $resultpath > logs/${exp}_raretest.log
echo -e "\nResults on rare words test set:"
tail -3 logs/${exp}_raretest.log

python test_char.py $freqtestdata $modelpath $resultpath > logs/${exp}_freqtest.log
echo -e "\nResults on frequent words test set:"
tail -3 logs/${exp}_freqtest.log
