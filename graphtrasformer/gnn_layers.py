
import math

import torch
import torch.nn as nn
from utils.utils import *

from torch_geometric.nn import GATConv,GCNConv,SAGEConv,GATv2Conv,TransformerConv,GINConv





class Geometric_GNN(nn.Module):
    def __init__(self,
                 gnn_type,
                 hidden_dim,
                 n_layers,
                 gnn_dropout,
                 use_super_node,
                 **kwargs
                 ):
        super(Geometric_GNN, self).__init__()
        self.use_super_node=use_super_node
        if gnn_type == 'GCN':
            self.gnn_list = torch.nn.ModuleList([GCNConv(hidden_dim, hidden_dim,**kwargs) for _ in range(n_layers)])
        elif gnn_type == 'SAGE':
            self.gnn_list = torch.nn.ModuleList([SAGEConv(hidden_dim, hidden_dim,**kwargs) for _ in range(n_layers)])
        elif gnn_type == 'GAT':
            self.n_heads=4 #just default
            self.gnn_list = torch.nn.ModuleList([GATConv(hidden_dim, int(hidden_dim / self.n_heads),
                                                              heads=self.n_heads, dropout=0,**kwargs) for _ in range(n_layers)])
        elif gnn_type == 'GATv2':
            self.n_heads=4
            self.gnn_list = torch.nn.ModuleList([GATv2Conv(hidden_dim, int(hidden_dim / self.n_heads),
                                                                heads=self.n_heads, dropout=0,**kwargs) for _ in range(n_layers)])

        elif gnn_type == 'GIN':
            gin_nn = nn.Sequential(nn.Linear(hidden_dim,hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim,hidden_dim),
                                   nn.ReLU())
            self.gnn_list = torch.nn.ModuleList([GINConv(nn=gin_nn) for _ in range (n_layers)])
        else:
            raise NotImplementedError('no such gnn type')
        self.dropout = nn.Dropout(p=gnn_dropout)
        self.act_fn = nn.ReLU()


    def forward(self, batched_data, x):
        edge_index ,x_mask= batched_data['edge_index'],batched_data['x_mask'].bool().reshape(-1)

        res = x.clone()


        x = x[:,int(self.use_super_node):,:].reshape(-1,x.shape[-1])
        x_zeros = torch.zeros(x.shape,device=x.device)
        x_graph_batch = x[x_mask]


        for layer in self.gnn_list:
            x_graph_batch = self.dropout(self.act_fn(layer(x_graph_batch,edge_index)))


        x_zeros[x_mask] = x_graph_batch

        res[:,int(self.use_super_node):,:]+=x_zeros.reshape(res.shape[0],-1,x.shape[-1])
        return res







