import copy
from typing import Optional

import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.data import Data, Batch
from torch_geometric.data import Dataset
from sklearn.model_selection import train_test_split
from typing import List
import torch
import numpy as np

from ..wrapper import preprocess_item
from .. import algos

import copy
from functools import lru_cache

from typing import Callable, List, NamedTuple, Optional, Tuple, Union

import torch
from torch import Tensor
from torch_sparse import SparseTensor


class EdgeIndex(NamedTuple):
    edge_index: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)


class Adj(NamedTuple):
    adj_t: SparseTensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        adj_t = self.adj_t.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return Adj(adj_t, e_id, self.size)





class GraphormerPYGDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        args,
        seed: int = 0,
        train_idx=None,
        valid_idx=None,
        test_idx=None,
        train_set=None,
        valid_set=None,
        test_set=None,
        x_norm_func=lambda x:x,
    ):
        self.args=args
        self.dataset = dataset
        if self.dataset is not None:
            self.num_data = len(self.dataset)
        self.seed = seed
        self.x_norm_func=x_norm_func
        if train_idx is None and train_set is None:
            train_valid_idx, test_idx = train_test_split(
                np.arange(self.num_data),
                test_size=self.num_data // 10,
                random_state=seed,
            )
            train_idx, valid_idx = train_test_split(
                train_valid_idx, test_size=self.num_data // 5, random_state=seed
            )
            self.train_idx = torch.from_numpy(train_idx)
            self.valid_idx = torch.from_numpy(valid_idx)
            self.test_idx = torch.from_numpy(test_idx)
            self.train_data = self.index_select(self.train_idx)
            self.valid_data = self.index_select(self.valid_idx)
            self.test_data = self.index_select(self.test_idx)
        elif train_set is not None:
            self.num_data = len(train_set) + len(valid_set) + len(test_set)
            self.train_data = self.create_subset(train_set)
            self.valid_data = self.create_subset(valid_set)
            self.test_data = self.create_subset(test_set)
            self.train_idx = None
            self.valid_idx = None
            self.test_idx = None
        else:
            self.num_data = len(train_idx) + len(valid_idx) + len(test_idx)
            self.train_idx = train_idx
            self.valid_idx = valid_idx
            self.test_idx = test_idx
            self.train_data = self.index_select(self.train_idx)
            self.valid_data = self.index_select(self.valid_idx)
            self.test_data = self.index_select(self.test_idx)
        self.__indices__ = None

    def index_select(self, idx):
        dataset = copy.copy(self)
        dataset.dataset = self.dataset.index_select(idx)
        if isinstance(idx, torch.Tensor):
            dataset.num_data = idx.size(0)
        else:
            dataset.num_data = idx.shape[0]
        dataset.__indices__ = idx
        dataset.train_data = None
        dataset.valid_data = None
        dataset.test_data = None
        dataset.train_idx = None
        dataset.valid_idx = None
        dataset.test_idx = None
        return dataset

    def create_subset(self, subset):
        dataset = GraphormerPYGDataset(subset,seed=self.seed,args=self.args)
        dataset.train_data = None
        dataset.valid_data = None
        dataset.test_data = None
        dataset.train_idx = None
        dataset.valid_idx = None
        dataset.test_idx = None
        return dataset


    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.dataset[idx]
            item.idx = idx
            item.y=item.y.reshape(1, -1) if item.y.shape[-1] > 1 else item.y.reshape(-1)

            return preprocess_item(item, self.x_norm_func, args=self.args)
        else:
            raise TypeError("index to a GraphormerPYGDataset can only be an integer.")

    def __len__(self):
        return self.num_data



class Graphtrans_Sampling_Dataset(Dataset):#shadowhop sampling
    def __init__(self,
                 data,
                 node_idx,
                 depth: int, num_neighbors: int,
                 replace: bool = False,
                 x_norm_func = lambda x:x,
                 args=None
                 ):

        self.data = data#copy.copy(data)
        self.depth = depth
        self.num_neighbors = num_neighbors
        self.replace = replace
        self.x_norm_func = x_norm_func
        self.args=args

        if data.edge_index is not None:
            self.is_sparse_tensor = False
            row, col = data.edge_index.cpu()
            self.adj_t = SparseTensor(
                row=row, col=col, value=torch.arange(col.size(0)),
                sparse_sizes=(data.num_nodes, data.num_nodes)).t()
        else:
            self.is_sparse_tensor = True
            self.adj_t = data.adj_t.cpu()

        if node_idx is None:
            node_idx = torch.arange(self.adj_t.sparse_size(0))
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)
        self.node_idx = node_idx
        self.num_data = len(self.node_idx)


    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        n_id = self.node_idx[idx]

        rowptr, col, value = self.adj_t.csr()
        out = torch.ops.torch_sparse.ego_k_hop_sample_adj(
            rowptr, col, n_id, self.depth, self.num_neighbors, self.replace)
        rowptr, col, n_id, e_id, ptr, root_n_id = out

        adj_t = SparseTensor(rowptr=rowptr, col=col,
                             value=value[e_id] if value is not None else None,
                             sparse_sizes=(n_id.numel(), n_id.numel()),
                             is_sorted=True)

        batch = Batch(batch=torch.ops.torch_sparse.ptr2ind(ptr, n_id.numel()),
                      ptr=ptr)
        batch.root_n_id = root_n_id

        if self.is_sparse_tensor:
            batch.adj_t = adj_t
        else:
            row, col, e_id = adj_t.t().coo()
            batch.edge_index = torch.stack([row, col], dim=0)

        for k, v in self.data:
            if k in ['edge_index', 'adj_t', 'num_nodes']:
                continue
            if k == 'y' and v.size(0) == self.data.num_nodes:
                batch[k] = v[n_id][root_n_id]
            elif isinstance(v, Tensor) and v.size(0) == self.data.num_nodes:
                batch[k] = v[n_id]
            elif isinstance(v, Tensor) and v.size(0) == self.data.num_edges:
                batch[k] = v[e_id]
            else:
                batch[k] = v

        item = batch
        item.idx = self.node_idx[idx]
        return preprocess_item(item,x_norm_func=self.x_norm_func,args=self.args)

    def __len__(self):
        return self.num_data




class Graphtrans_Sampling_Dataset_v2(Dataset):#sage sampling +induced subgraph
    def __init__(self,
                 data,
                 node_idx,
                 depth: int,
                 num_neighbors,
                 replace: bool = False,
                 x_norm_func = lambda x:x,
                 args=None
                 ):

        self.data = copy.copy(data)
        self.depth = depth
        if isinstance(num_neighbors,int):
            self.num_neighbors = [num_neighbors]+(depth-1)*[1]
        self.replace = replace
        self.x_norm_func = x_norm_func
        self.args=args


        if data.edge_index is not None:
            self.is_sparse_tensor = False
            row, col = data.edge_index.cpu()
            self.adj_t = SparseTensor(
                row=row, col=col, value=torch.arange(col.size(0)),
                sparse_sizes=(data.num_nodes, data.num_nodes)).t()
        else:
            self.is_sparse_tensor = True
            self.adj_t = data.adj_t.cpu()


        if node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)
        self.node_idx = node_idx
        self.num_data = len(self.node_idx)


    def __getitem__(self, idx):

        n_id = self.node_idx[idx].reshape(1)
        root_n_id=0
        hop_node_nums = [n_id.shape[0]]

        for size in self.num_neighbors:
            adj_t, n_id = self.adj_t.sample_adj(n_id, size, replace=False)
            hop_node_nums.append(n_id.shape[0])
        n_hops = torch.ones(n_id.shape)+len(self.num_neighbors)
        for i,hop_offset in enumerate(hop_node_nums):
            n_hops[:hop_offset]-=1

        adj_t,_ = self.adj_t.saint_subgraph(n_id)
        row, col, e_id = adj_t.t().coo()
        edge_index = torch.stack([row, col])


        item = Data(x = self.data.x[n_id],edge_index=edge_index)
        for k, v in self.data:
            if k in ['edge_index', 'adj_t', 'num_nodes']:
                continue
            if k == 'y' and v.size(0) == self.data.num_nodes:
                item[k] = v[n_id][root_n_id].reshape(1)
            elif isinstance(v, Tensor) and v.size(0) == self.data.num_nodes:
                item[k] = v[n_id]
            elif isinstance(v, Tensor) and v.size(0) == self.data.num_edges:
                item[k] = v[e_id]
            else:
                item[k] = v
        item.root_n_id = root_n_id

        item.idx = self.node_idx[idx]
        item.n_hops = n_hops
        return preprocess_item(item,x_norm_func=self.x_norm_func,args=self.args)

    def __len__(self):
        return self.num_data


