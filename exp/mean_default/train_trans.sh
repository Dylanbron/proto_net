#! /bin/bash
T=`date +%m%d%H%M`
ROOT=../../
export PYTHONPATH=$ROOT:$PYTHONPATH
log_dir='log'
if [ ! -d "$log_dir" ]; then
	mkdir $log_dir
fi
python $ROOT/train_rgb_aug.py --gpu 0  --save-path './save/' |tee ./log/train.log.$T

