import copy
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


class relationDataset(Dataset):
    def __init__(self, train_file, entity2id, relation2id):
        self.relation = defaultdict(lambda : 0)
        self.entity = defaultdict(lambda : 0)
        self.head = []
        self.tail = []
        self.rel = []
        self.triple = defaultdict(list)
        with open(entity2id, 'r') as f:
            self.entity_num = int(f.readline().strip())
            for line in f:
                ent, idx = line.strip().split('\t')
                self.entity[ent] = int(idx)
        with open(relation2id, 'r') as f:
            self.relation_num = int(f.readline().strip())
            for line in f:
                rel, idx = line.strip().split('\t')
                self.relation[rel] = int(idx)

        with open(train_file, 'r') as f:
            self.triple_num = int(f.readline().strip())
            for line in f:
                head, tail, rel = line.strip().split()
                self.head.append(int(head))
                self.tail.append(int(tail))
                self.rel.append(int(rel))
                self.triple[int(rel)].append([int(head), int(tail)])
            self.head = np.array(self.head)
            self.rel = np.array(self.rel)
            self.tail = np.array(self.tail)
        
            
        print("Relation nums:%d"%len(self.relation))
        print("Entity nums:%d"%len(self.entity))
        print("Triple nums:%d"%self.head.shape[0])

    def __len__(self):
        return self.head.shape[0]

    def __getitem__(self, index):
        return self.head[index], self.rel[index], self.tail[index], 1