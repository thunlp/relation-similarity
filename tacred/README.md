The codes are based on https://github.com/yuhaozhang/tacred-relation.



First, run the codes of preparation step as mentioned in https://github.com/yuhaozhang/tacred-relation. 

Then you can use the following code to train a relation extraction model on tacred.

```
CUDA_VISIBLE_DEVICES=0 python train.py --data_dir dataset/tacred --vocab_dir dataset/vocab --id 0 --init_temp $init_temp --alpha $alpha --end_temp $end_temp --var_temp
```

And use the following code to evaluate the model

```
CUDA_VISIBLE_DEVICES=0 python eval.py saved_models/$id --dataset test
```

