from data.dataset import *
from torch_geometric.datasets import *
import torch.nn as nn
#from ogb.lsc.pcqm4m_pyg import PygPCQM4MDataset
from tqdm import tqdm

from pcqm4mv2_pyg import PygPCQM4Mv2Dataset
from pcqm4m_pyg import PygPCQM4MDataset
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import Evaluator
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric
from torch_geometric.loader import NeighborLoader, ShaDowKHopSampler
from data.pyg_datasets.pyg_dataset import GraphormerPYGDataset, Graphtrans_Sampling_Dataset,Graphtrans_Sampling_Dataset_v2
#12000
import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.data import Data, Batch
from torch_geometric.data import Dataset
from sklearn.model_selection import train_test_split
from typing import List
import torch
import numpy as np
from data.wrapper import preprocess_item


import copy
from functools import lru_cache

def get_loss_and_metric(data_name):

    if data_name in ['ZINC','pcqm4mv2','QM7','QM9','ZINC-full']:
        loss = nn.L1Loss(reduction='mean')
        metric =  nn.L1Loss(reduction='mean')
        task_type='regression'
        metric_name = 'MAE'

    elif data_name in ['UPFD']:
        loss = nn.BCEWithLogitsLoss(reduction='mean')
        metric = load_metric("accuracy")
        task_type='binary_classification'
        metric_name='accuracy'

    elif data_name in ["ogbg-molhiv"]:
        loss = nn.BCEWithLogitsLoss(reduction='mean')
        metric = Evaluator(name=data_name)
        task_type='binary_classification'
        metric_name='ROC-AUC'

    elif data_name in ['flickr','ogbn-products','ogbn-arxiv']:
        loss = nn.CrossEntropyLoss(reduction='mean')
        metric = load_metric('accuracy')
        task_type='multi_classification'
        metric_name='accuracy'
    elif data_name in ["ogbg-molpcba"]:
        loss = nn.BCEWithLogitsLoss(reduction='mean')

        metric = Evaluator(name=data_name)
        task_type='multi_binary_classification'
        metric_name='AP'

    else:
        raise ValueError('no such dataset')

    return loss, metric, task_type,metric_name



def normalization(data_list,mean,std):
    for i in tqdm(range(len(data_list))):
        data_list[i] = (data_list[i].x-mean)/std
    return data_list


def get_graph_level_dataset(name,param=None,seed=1024,set_default_params=False,args=None):

    path = 'dataset/'+name
    print(path)
    train_set = None
    val_set = None
    test_set = None
    inner_dataset = None
    train_idx=None
    val_idx=None
    test_idx=None

    #graph regression
    if name=='ZINC':#250,000 molecular graphs with up to 38 heavy atoms
        train_set = ZINC(path,subset=True,split='train')
        val_set = ZINC(path,subset=True,split='val')
        test_set = ZINC(path,subset=True,split='test')
        args.node_feature_type='cate'
        args.num_class =1

        args.eval_steps=1000
        args.save_steps=1000

        args.greater_is_better = False

        args.warmup_steps=40000
        args.max_steps=400000
    elif name == 'ZINC-full':  # 250,000 molecular graphs with up to 38 heavy atoms
        train_set = ZINC(path, subset=False, split='train')
        val_set = ZINC(path, subset=False, split='val')
        test_set = ZINC(path, subset=False, split='test')
        args.node_feature_type = 'cate'
        args.num_class = 1

        args.eval_steps = 1000
        args.save_steps = 1000

        args.greater_is_better = False

        args.warmup_steps = 40000
        args.max_steps = 400000
        #args.node_level_modules = ()
        #args.attn_level_modules = ()
    elif name == "ogbg-molpcba":
        inner_dataset = PygGraphPropPredDataset(name)
        idx_split = inner_dataset.get_idx_split()
        train_idx = idx_split["train"]
        val_idx = idx_split["valid"]
        test_idx = idx_split["test"]
        args.node_feature_type = 'cate'
        args.num_class = 128

        args.eval_steps = 2000
        args.save_steps = 2000

        args.greater_is_better = True

        args.warmup_steps = 40000
        args.max_steps = 1000000


        #args.warmup_steps = 40000
        #args.max_steps = 800000

    elif name == "ogbg-molhiv":
        inner_dataset = PygGraphPropPredDataset(name)
        idx_split = inner_dataset.get_idx_split()
        train_idx = idx_split["train"]
        val_idx = idx_split["valid"]
        test_idx = idx_split["test"]
        args.node_feature_type = 'cate'
        args.num_class = 1

        args.eval_steps = 1000
        args.save_steps = 1000

        args.greater_is_better = True

        args.warmup_steps = 40000
        args.max_steps = 1200000




    elif name=='UPFD' and param in ('politifact', 'gossipcop'):
        train_set = UPFD(path,param,'bert',split='train')
        val_set = UPFD(path,param,'bert',split='val')
        test_set = UPFD(path,param,'bert',split='test')
        args.learning_rate=1e-5
        args.node_feature_type='dense'
        args.node_feature_dim=768

        args.greater_is_better = True



    else:
        raise ValueError('no such dataset')


    #return train_set,val_set,test_set,dataset
    dataset = GraphormerPYGDataset(
        dataset=inner_dataset,
        train_idx=train_idx,
        valid_idx=val_idx,
        test_idx=test_idx,
        train_set=train_set,
        valid_set=val_set,
        test_set=test_set,
        seed=seed,
        args=args
                )
    return dataset.train_data,dataset.valid_data,dataset.test_data, inner_dataset


def get_node_level_dataset(name,param=None,args=None):
    path = 'dataset/' + name
    print(path)


    #args.use_super_node=False

    if args.sampling_algo=='shadowkhop':
        args.num_neighbors=10
    elif args.sampling_algo=='sage':
        args.num_neighbors=50

    #node classification  tranductive/inductive  link prediction
    if name in ['cora','citeseer','dblp','pubmed']:
        dataset = CitationFull(f'dataset/{name}',name)
        N = dataset.data.x.shape[0]


    elif name =='flickr':
        dataset = Flickr(path)
        x_norm_func = lambda x:x #

        args.node_feature_dim=500
        args.node_feature_type='dense'
        args.num_class =7

        args.encoder_normalize_before =True
        args.apply_graphormer_init =True
        args.greater_is_better = True

        args.warmup_steps=2000
        args.max_steps=100000

        train_idx = dataset.data.train_mask.nonzero().squeeze()
        valid_idx = dataset.data.val_mask.nonzero().squeeze()
        test_idx = dataset.data.test_mask.nonzero().squeeze()


    elif name=='ogbn-products':

        dataset = PygNodePropPredDataset(name='ogbn-products')
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

        x_norm_func = lambda x:x

        args.node_feature_dim=100
        args.node_feature_type='dense'
        args.num_class =47

        args.encoder_normalize_before =True
        args.apply_graphormer_init =True
        args.greater_is_better = True

        args.warmup_steps=10000
        args.max_steps=400000


    elif name =='ogbn-arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv')
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

        x_norm_func = lambda x:x

        args.node_feature_dim=128
        args.node_feature_type='dense'
        args.num_class =40

        args.encoder_normalize_before =True
        args.apply_graphormer_init =True
        args.greater_is_better = True

        args.warmup_steps=10000
        args.max_steps=800000


    else:
        raise ValueError('no such dataset')


    if args.sampling_algo=='shadowkhop':
        Sampling_Dataset = Graphtrans_Sampling_Dataset
    elif args.sampling_algo=='sage':
        Sampling_Dataset = Graphtrans_Sampling_Dataset_v2
        args.num_neighbors=50


    train_set = Sampling_Dataset(dataset.data,
                                          node_idx=train_idx,
                                          depth=args.depth,
                                          num_neighbors=args.num_neighbors,
                                          replace=False,
                                          x_norm_func=x_norm_func,
                                            args=args)
    valid_set = Sampling_Dataset(dataset.data,
                                          node_idx=valid_idx,
                                          depth=args.depth,
                                          num_neighbors=args.num_neighbors,
                                          replace=False,
                                          x_norm_func=x_norm_func,
                                            args=args)
    test_set = Sampling_Dataset(dataset.data,
                                          node_idx=test_idx,
                                          depth=args.depth,
                                          num_neighbors=args.num_neighbors,
                                          replace=False,
                                          x_norm_func=x_norm_func,
                                           args=args)

    return train_set,valid_set,test_set, dataset, args







#just test
if __name__=='__main__':
    #name='QM9'

    #heterogeneous
    pass



