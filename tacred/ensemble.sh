#!/bin/bash

# This is an example script of training and running model ensembles.

# train 5 models with different seeds
temp=1;
for alpha in 1 2 3 4 5 6 7 8 9 10;
do
        CUDA_VISIBLE_DEVICES=0 python train.py --seed 0 --id 05 --init_temp $temp --alpha $alpha
        CUDA_VISIBLE_DEVICES=0 python train.py --seed 1 --id 06 --init_temp $temp --alpha $alpha
        CUDA_VISIBLE_DEVICES=0 python train.py --seed 2 --id 07 --init_temp $temp --alpha $alpha
        CUDA_VISIBLE_DEVICES=0 python train.py --seed 3 --id 08 --init_temp $temp --alpha $alpha
        CUDA_VISIBLE_DEVICES=0 python train.py --seed 4 --id 09 --init_temp $temp --alpha $alpha

# evaluate on test sets and save prediction files
        CUDA_VISIBLE_DEVICES=0 python eval.py saved_models/05 --out saved_models/out/test_5.pkl
        CUDA_VISIBLE_DEVICES=0 python eval.py saved_models/06 --out saved_models/out/test_6.pkl
        CUDA_VISIBLE_DEVICES=0 python eval.py saved_models/07 --out saved_models/out/test_7.pkl
        CUDA_VISIBLE_DEVICES=0 python eval.py saved_models/08 --out saved_models/out/test_8.pkl
        CUDA_VISIBLE_DEVICES=0 python eval.py saved_models/09 --out saved_models/out/test_9.pkl

# run ensemble
        ARGS=""
        for id in 5 6 7 8 9; do
            OUT="saved_models/out/test_${id}.pkl"
        ARGS="$ARGS $OUT"
        done
        python ensemble.py --dataset test --temp $temp --alpha $alpha $ARGS
done