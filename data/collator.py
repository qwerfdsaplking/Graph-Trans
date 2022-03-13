# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch_geometric.data import Batch,Data

def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_pos_emb_unsqueeze(x, padlen):
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_adj_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


def collator(items, args):

    max_node = args.max_node
    multi_hop_max_dist = args.multi_hop_max_dist
    spatial_pos_max = args.spatial_pos_max

    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [
        (
            item.idx,
            item.attn_bias,
            item.attn_edge_type,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.edge_input,
            item.y,
            item.adj,
            item.adj_norm,
            item.edge_index,
            item.eig_pos_emb,
            item.svd_pos_emb,
            item.root_n_id
        )
        for item in items
    ]
    (
        idxs,
        attn_biases,
        attn_edge_types,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_inputs,
        ys,
        adjs,
        adj_norms,
        edge_indexs,
        eig_pos_embs,
        svd_pos_embs,
        root_n_ids
    ) = zip(*items)

    for i, _ in enumerate(attn_biases):
        attn_biases[i][int(args.use_super_node):, int(args.use_super_node):][spatial_poses[i] >= spatial_pos_max] = float("-inf")
    max_node_num = max(i.size(0) for i in xs)
    ns = [x.size(0) for x in xs]
    x_mask = torch.zeros(len(xs),max_node_num)
    for i,n in enumerate(ns):
        x_mask[i,:n]=1



    y = torch.cat(ys)
    root_n_id = torch.tensor(root_n_ids)

    if args.node_feature_type=='cate':
        x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    else:
        x = torch.cat([pad_pos_emb_unsqueeze(i, max_node_num) for i in xs])


    if isinstance(edge_inputs[0],int):
        edge_input=None
        attn_edge_type=None
    else:
        max_dist = max(i.size(-2) for i in edge_inputs)
        edge_input = torch.cat(
            [pad_3d_unsqueeze(i[:, :, :multi_hop_max_dist, :], max_node_num, max_node_num, max_dist) for i in edge_inputs]
        )
        attn_edge_type = torch.cat(
            [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
        )

    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + int(args.use_super_node)) for i in attn_biases]
    )


    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees]) if not isinstance(in_degrees[0],int) else None
    adj = torch.cat([pad_adj_unsqueeze(a, max_node_num) for a in adjs])

    adj_norm = torch.cat([pad_adj_unsqueeze(a, max_node_num) for a in adj_norms]) if not isinstance(adj_norms[0], int) else None


    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    ) if not isinstance(spatial_poses[i],int) else None


    batch_edge_index = Batch.from_data_list([Data(edge_index=ei, num_nodes=ns[i]) for i, ei in enumerate(edge_indexs)]).edge_index if args.use_gnn_layers else None


    eig_pos_embs = torch.cat([pad_pos_emb_unsqueeze(i, max_node_num) for i in eig_pos_embs]) if not isinstance(eig_pos_embs[0],int) else None

    svd_pos_embs = torch.cat([pad_pos_emb_unsqueeze(i, max_node_num) for i in svd_pos_embs]) if not isinstance(svd_pos_embs[0],int) else None



    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        x=x,
        edge_input=edge_input,
        x_mask = x_mask,
        ns = torch.LongTensor(ns), #node number in each graph
        labels=y.squeeze(),
        adj = adj,
        adj_norm=adj_norm,
        edge_index = batch_edge_index,
        eig_pos_emb=eig_pos_embs,
        svd_pos_emb=svd_pos_embs,
        root_n_id=root_n_id
    )
