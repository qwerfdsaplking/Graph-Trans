# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from utils import utils
import torch
import numpy as np
from ogb.graphproppred import PygGraphPropPredDataset
#from ogb.lsc.pcqm4mv2_pyg import PygPCQM4Mv2Dataset
from functools import lru_cache
import pyximport
import torch.distributed as dist
from torch_geometric.utils import to_undirected,add_self_loops
pyximport.install(setup_args={"include_dirs": np.get_include()})
from . import algos
from utils.utils import *
from copy import deepcopy

@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(raw_item,x_norm_func,args):
    #raw_item=deepcopy(item)
    edge_attr, edge_index, x,y,idx = raw_item.edge_attr, raw_item.edge_index, raw_item.x,raw_item.y,raw_item.idx
    root_n_id=raw_item.root_n_id if 'root_n_id' in raw_item.to_dict().keys() else -1



    N = x.size(0)
    if args.node_feature_type=='cate':
        x = convert_to_single_emb(x)#????
    elif args.node_feature_type=='dense':
        x = x_norm_func(x)
    else:
        raise ValueError('node feature type error')

    # node adj matrix [N, N] bool
    #print(edge_index)
    try:
        edge_index = to_undirected(edge_index)
    except:
        print(edge_index)
        assert  1==2

    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    adj_w_sl = adj.clone()#adj with self loop
    adj_w_sl[torch.arange(N),torch.arange(N)]=1

    #positional bias
    if 'degree' in args.node_level_modules:
        in_degree = adj.long().sum(dim=1).view(-1)
    else:
        in_degree = 0

    if 'eig' in args.node_level_modules:
        if N<args.eig_pos_dim+1:
            #print(adj_w_sl,x,idx)
            eig_pos_emb =torch.zeros(N,args.eig_pos_dim)
        else:
            eigval,eigvec = utils.get_eig_dense(adj_w_sl)
            eig_idx = eigval.argsort()
            eigval,eigvec=eigval[eig_idx],eigvec[:,eig_idx]
            eig_pos_emb = eigvec[:,1:args.eig_pos_dim+1]

    else:
        eig_pos_emb = 0

    if 'svd' in args.node_level_modules:
        if N < args.svd_pos_dim:
            #print(adj_w_sl,x,idx)
            svd_pos_emb = torch.zeros(N,args.svd_pos_dim*2)
        else:
            pu,pv = utils.get_svd_dense(adj_w_sl,args.svd_pos_dim)
            svd_pos_emb = torch.cat([pu,pv],dim=-1)
    else:
        svd_pos_emb = 0

    #attention bias
    if 'nhop' in args.attn_level_modules:
        adj_norm = unweighted_adj_normalize_dense_batch(adj_w_sl)
    else:
        adj_norm = 0

    if 'spatial' in args.attn_level_modules or 'sdp' in args.attn_level_modules:
        # edge feature here
        if edge_attr is not None and 'sdp' in args.attn_level_modules:
            if len(edge_attr.size()) == 1:
                edge_attr = edge_attr[:, None]
            attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
            attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
                convert_to_single_emb(edge_attr) + 1
            )
        else:
            attn_edge_type=None

        shortest_path_result, path = algos.floyd_warshall(adj.numpy())
        max_dist = np.amax(shortest_path_result)


        if attn_edge_type is not None:
            edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
            edge_input = torch.from_numpy(edge_input).long()
        else:
            edge_input = 0
            attn_edge_type = 0

        spatial_pos = torch.from_numpy((shortest_path_result)).long()
    else:
        attn_edge_type=0
        spatial_pos=0
        edge_input=0


    #super node
    if args.use_super_node:
        attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token
    else:
        attn_bias = torch.zeros([N, N], dtype=torch.float)

    # combine
    #node features


    item = Data(x=x,
                edge_index=edge_index,
                y=y,
                attn_bias=attn_bias,
                in_degree=in_degree,
                out_degree=in_degree,
                eig_pos_emb=eig_pos_emb,
                svd_pos_emb=svd_pos_emb,
                spatial_pos=spatial_pos,
                attn_edge_type=attn_edge_type,
                edge_input=edge_input,
                adj=adj_w_sl,
                adj_norm=adj_norm,
                idx=idx,
                root_n_id=root_n_id
                )

    return item





class MyPygGraphPropPredDataset(PygGraphPropPredDataset):
    def download(self):
        if dist.get_rank() == 0:
            super(MyPygGraphPropPredDataset, self).download()
        dist.barrier()

    def process(self):
        if dist.get_rank() == 0:
            super(MyPygGraphPropPredDataset, self).process()
        dist.barrier()

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.get(self.indices()[idx])
        item.idx = idx
        item.y = item.y.reshape(-1)
        return preprocess_item(item)
