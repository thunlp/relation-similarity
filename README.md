# Relation Similarity

Using fact distribution to quantify the similarity between relations in knowledge bases and real world. 

If you use the code, pleace cite the following [paper](https://www.aclweb.org/anthology/P19-1278):

```
@inproceedings{chen2019quantifying,
  title={Quantifying Similarity between Relations with Fact Distribution},
  author={Chen, Weize and Zhu, Hao and Han, Xu and Liu, Zhiyuan and Sun, Maosong},
  booktitle={Proceedings of the 57th Conference of the Association for Computational Linguistics},
  pages={2882--2894},
  year={2019}
}
```

## Requirements

+ Python 3 (tested on 3.6.7)
+ PyTorch (tested on 1.0.0)
+ Numpy (tested on 1.16.0)
+ Tqdm (tested on 4.30.0)
+ TensorboardX (tested on 1.6)

### Installation

```
pip install -r requirements.txt
```

## Training

You can use the following code to train a model that is capable of modeling fact distribution.

```
python train.py --input $Directory_to_your_own_dataset --output ./checkpoint -ent_pretrain -rel_pretrain 
```

If you want to train the model on the provided wikipedia dataset, you have to decompress the pretrained entity embedding file first.

Note that if you choose to add "-ent_pretrain" and "-rel_pretrain", ensure that you have pretrained embedding file "entity2vec.vec" and "relation2vec.vec" in your input directory. In our paper, the two pretrained embedding files are produced by running TransE on the dataset. We use the TransE implementation in [OpenKE](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch).

## Yield relation similarity

You can use the following code to yield similarity between relations after training the model.

```
python get_sim.py --model_path ./checkpoint --input ./data/wikipedia/ --output ./result
```

Two files will be produced, "kl_prob.txt" and "kl_prob.json". They are the same except the file format. The i-th line contains the KL divergence between i-th relation and other relations. 

## Hyper-parameters

All the default hyper-parameters are the ones used in the paper.

## Relation prediction and relation extraction

We perform relation prediction using the implementation of https://github.com/thunlp/OpenKE/tree/OpenKE-alpha. You can run "example_train_transe.py" to reproduce the result.

As for relation extraction, we put the code in "tacred" directory. Due to copyright issues, we did not publish the dataset.
