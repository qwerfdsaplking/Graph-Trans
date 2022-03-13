import numpy as np
import torch
import scipy.sparse as sp
from numpy.linalg import inv
import pickle

from torch_geometric.datasets import *

import torch
import numpy as np
from torch_sparse.matmul import matmul
from torch_sparse import SparseTensor


c = 0.15
k = 5


def adj_normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def get_intimacy_matrix(edges,n):
    edges= np.array(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n,n),
                        dtype=np.float32)
    print('normalize')
    adj_norm = adj_normalize(adj)
    print('inverse')
    eigen_adj = c * inv((sp.eye(adj.shape[0]) - (1 - c) * adj_norm).toarray())

    return eigen_adj


def adj_normalize_sparse(mx):
    mx=mx.to(device)
    rowsum = mx.sum(1)
    r_inv =rowsum.pow(-0.5).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = SparseTensor(row = torch.arange(n).to(device),col=torch.arange(n).to(device),value=r_inv, sparse_sizes=(n,n))
    nr_mx = matmul(matmul(r_mat_inv,mx),r_mat_inv)
    return nr_mx

def get_intimacy_matrix_sparse(edges,n):
    adj = SparseTensor(row=edges[0], col=edges[1], value=torch.ones(edges.shape[1]), sparse_sizes=(n, n))
    adj_norm = adj_normalize_sparse(adj)
    return adj_norm

def get_svd_dense(mx,q=3):
    mx = mx.float()
    u,s,v = torch.svd_lowrank(mx,q=q)
    s=torch.diag(s)
    pu = u@s.pow(0.5)
    pv = v@s.pow(0.5)
    return pu,pv


def unweighted_adj_normalize_dense_batch(adj):
    adj = (adj+adj.transpose(-1,-2)).bool().float()
    adj = adj.float()
    rowsum = adj.sum(-1)
    r_inv = rowsum.pow(-0.5)
    r_mat_inv = torch.diag_embed(r_inv)
    nr_adj = torch.matmul(torch.matmul(r_mat_inv,adj),r_mat_inv)
    return nr_adj


def get_eig_dense(adj):
    adj = adj.float()
    rowsum = adj.sum(1)
    r_inv =rowsum.pow(-0.5)
    r_mat_inv = torch.diag(r_inv)
    nr_adj = torch.matmul(torch.matmul(r_mat_inv,adj),r_mat_inv)
    graph_laplacian = torch.eye(adj.shape[0])-nr_adj
    L,V = torch.eig(graph_laplacian,eigenvectors=True)
    return L.T[0],V



def check_checkpoints(output_dir):
    import os
    import shutil
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        for file in files:
            if 'checkpoint' in file:

                return True
        print('remove ',output_dir)
        shutil.rmtree(output_dir)
    return False


if __name__=='__main__':
    #just test

    device = torch.device('cuda',0)

    data = Flickr('dataset/flickr')

    edges= data.data.edge_index
    n=data.data.x.shape[0]


    adj = SparseTensor(row=edges[0], col=edges[1], value=torch.ones(edges.shape[1]), sparse_sizes=(n, n))
    nr_adj = adj_normalize_sparse(adj)

    pu,pv= get_svd_dense(nr_adj.to_torch_sparse_coo_tensor(),q=10)


    adj= (torch.randn(10,10)>0).float()
    L,V = get_eig_dense(adj)
