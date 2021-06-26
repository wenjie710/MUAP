#!/usr/bin/env bash

path='tmp_rank1/'
net='densenet_'
npy='.npy'
for mode in 'class' 'base' 'tv';
do
    dataset='market/'
    python tools/test_market_paperorder_rank1.py --config_file=configs/attack.yml MODEL.DEVICE_ID "('1')" ATTACK.ATTACK_PATH $path$dataset$net$mode$npy
    dataset='duke/'
    python tools/test_duke_paperorder_rank1.py --config_file=configs/attack.yml MODEL.DEVICE_ID "('1')" ATTACK.ATTACK_PATH $path$dataset$net$mode$npy
    dataset='msmt/'
    python tools/test_market_paperorder_rank1.py --config_file=configs/attack.yml MODEL.DEVICE_ID "('1')" ATTACK.ATTACK_PATH $path$dataset$net$mode$npy
    python tools/test_duke_paperorder_rank1.py --config_file=configs/attack.yml MODEL.DEVICE_ID "('1')" ATTACK.ATTACK_PATH $path$dataset$net$mode$npy
done


#python tools/test_market_paperorder_rank1.py --config_file=configs/attack.yml MODEL.DEVICE_ID "('6')"  ATTACK.ATTACK_PATH "('zeros.npy')"
