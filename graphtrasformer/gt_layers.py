import math

import torch
import torch.nn as nn
from utils.utils import *

def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class Eig_Embedding(nn.Module):
    def __init__(self, eig_dim, hidden_dim):
        super(Eig_Embedding,self).__init__()
        self.embeddings = nn.Linear(eig_dim,hidden_dim)
    def forward(self, batched_data):
        pos = batched_data['eig_pos_emb']
        sign = torch.randn(1)[0]>0
        sign = 1 if sign else -1
        return self.embeddings(pos * sign)

class SVD_Embedding(nn.Module):
    def __init__(self, svd_dim, hidden_dim):
        super(SVD_Embedding,self).__init__()
        self.svd_dim=svd_dim
        self.embeddings = nn.Linear(svd_dim*2,hidden_dim)
    def forward(self, batched_data):
        pos = batched_data['svd_pos_emb']
        sign = torch.randn(1)[0]>0
        sign = 1 if sign else -1
        pos_u = pos[:,:,:self.svd_dim]*sign
        pos_v = pos[:,:,self.svd_dim:]*(-sign)
        pos = torch.cat([pos_u,pos_v],dim=-1)
        return self.embeddings(pos)


class WL_Role_Embedding(nn.Module):
    def __init__(self, max_index, hidden_dim):
        super(WL_Role_Embedding,self).__init__()
        self.embeddings = nn.Linear(max_index,hidden_dim)
    def forward(self, batched_data):
        pos = batched_data['wl_role_ids']
        return self.embeddings(pos)

class Inti_Pos_Embedding(nn.Module):
    def __init__(self, max_index, hidden_dim):
        super(Inti_Pos_Embedding,self).__init__()
        self.embeddings = nn.Linear(max_index,hidden_dim)
    def forward(self, batched_data):
        pos = batched_data['init_pos_ids']
        return self.embeddings(pos)

class Hop_Dis_Embedding(nn.Module):
    def __init__(self, max_index, hidden_dim):
        super(Hop_Dis_Embedding,self).__init__()
        self.embeddings = nn.Linear(max_index,hidden_dim)
    def forward(self, batched_data):
        pos = batched_data['hop_dis_ids']
        return self.embeddings(pos)

class DegreeEncoder(nn.Module):
    def __init__(self,
                 num_in_degree,
                 num_out_degree,
                 hidden_dim,
                 n_layers #for parameter initialization
                 ):
        super(DegreeEncoder, self).__init__()
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        in_degree, out_degree = (
            batched_data["in_degree"],
            batched_data["out_degree"],
        )
        return self.in_degree_encoder(in_degree)+self.out_degree_encoder(out_degree)


class AddSuperNode(nn.Module):
    def __init__(self, hidden_dim):
        super(AddSuperNode, self).__init__()
        self.graph_token = nn.Embedding(1, hidden_dim)

    def forward(self, node_feature):
        n_graph = node_feature.size()[0]
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        return graph_node_feature





class NodeFeatureEncoder(nn.Module):
    def __init__(
            self,
            feat_type,
            hidden_dim,
            n_layers,
            num_atoms=None,
            feat_dim=None
    ):
        super(NodeFeatureEncoder, self).__init__()

        self.feat_type = feat_type

        if feat_type=='dense' and feat_dim is not None:#dense feature
            self.feature_encoder = nn.Linear(feat_dim, hidden_dim)
        elif feat_type=='cate' and num_atoms is not None:#cate feature
            # 1 for graph token
            self.feature_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        else:
            raise ValueError('conflict feature type')

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        x=batched_data["x"]
        if self.feat_type=='cate':#
            node_feature = self.feature_encoder(x).sum(dim=-2)  # [n_graph, n_node, n_hidden]
        else:
            node_feature = self.feature_encoder(x)

        return node_feature


def getAttnMasks(batched_data,attn_mask_modules,use_super_node,num_heads):
    adj = batched_data['adj'].bool().float()

    attn_mask = torch.ones(adj.shape[0], num_heads,adj.shape[1] + int(use_super_node),
                                 adj.shape[2] + int(use_super_node)).to(adj.device)
    if attn_mask_modules == '1hop':
        adjs = adj.unsqueeze(1).expand(-1,num_heads,-1,-1).bool().float()
        attn_mask[:,:,int(use_super_node):,int(use_super_node):] = adjs


    if attn_mask_modules == 'nhop':
        multi_hop_adjs = torch.cat([torch.matrix_power(adj, i + 1).unsqueeze(1) for i in range(num_heads)],
                                   dim=1).bool().float()
        attn_mask[:,:, int(use_super_node):, int(use_super_node):] = multi_hop_adjs

    return attn_mask


class GraphAttnHopBias(nn.Module):
    def __init__(
        self,
        num_heads,
        n_hops,
        use_super_node
    ):
        super(GraphAttnHopBias, self).__init__()
        self.num_heads = num_heads
        self.use_super_node=use_super_node
        self.hop_bias = nn.Parameter(torch.randn(n_hops,num_heads))
        self.n_hops = n_hops

    def forward(self, batched_data):
        x, adj, attn_bias = (
            batched_data["x"],
            batched_data['adj_norm'],
            batched_data['attn_bias']
        )


        adj_n_hops_bias = torch.ones(adj.shape[0],adj.shape[1]+int(self.use_super_node),
                                     adj.shape[2]+int(self.use_super_node),self.n_hops).to(x.device)
        adj_list = [torch.matrix_power(adj,i+1).unsqueeze(-1) for i in range(self.n_hops)]
        adj_n_hops = torch.cat(adj_list,dim=-1)# n_graph, n_node, n_node, n_hops
        adj_n_hops_bias[:,int(self.use_super_node):,int(self.use_super_node):,:] = adj_n_hops
        adj_n_hops_bias = torch.matmul(adj_n_hops_bias,self.hop_bias).permute(0, 3, 1, 2)

        return adj_n_hops_bias# [n_graph, n_head, n_node+1, n_node+1]




class GraphAttnSpatialBias(nn.Module):#refer to Graphormer
    def __init__(
        self,
        num_heads,
        num_spatial,
        n_layers,
        use_super_node
    ):
        super(GraphAttnSpatialBias, self).__init__()
        self.num_heads = num_heads
        self.use_super_node = use_super_node

        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)

        if use_super_node:
            self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        attn_bias, spatial_pos, x = (
            batched_data["attn_bias"],#[n_graph, n_node+1, n_node+1]
            batched_data["spatial_pos"],#[n_graph, n_node, n_node]
            batched_data["x"],
        )

        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, int(self.use_super_node):, int(self.use_super_node):] = graph_attn_bias[:, :, int(self.use_super_node):, int(self.use_super_node):] + spatial_pos_bias

        # reset spatial pos here
        if self.use_super_node:
            t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
            graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
            graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset  pad -inf

        return graph_attn_bias# [n_graph, n_head, n_node+1, n_node+1]



class GraphAttnEdgeBias(nn.Module): #refer to Graphormer
    """
    Compute attention bias for each head.  We do not need to consider super node in this module.
    """
    def __init__(
        self,
        num_heads,
        num_edges,
        num_edge_dis,
        edge_type,
        multi_hop_max_dist,
        n_layers,
    ):
        super(GraphAttnEdgeBias, self).__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist
        #probably some issues here
        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
        self.edge_type = edge_type
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(
                num_edge_dis * num_heads * num_heads, 1
            )

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        attn_bias, spatial_pos, x = (
            batched_data["attn_bias"],
            batched_data["spatial_pos"],
            batched_data["x"],
        )
        edge_input, attn_edge_type = (
            batched_data["edge_input"],
            batched_data["attn_edge_type"],
        )

        n_graph, n_node = x.size()[:2]


        if attn_edge_type is None:
            edge_input = torch.zeros(n_graph, self.num_heads, n_node, n_node).to(x.device)
            return edge_input

        # edge feature
        if self.edge_type == "multi_hop":
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, x > 1 to x - 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]
            # [n_graph, n_node, n_node, max_dist, n_head]
            edge_input = self.edge_encoder(edge_input).mean(-2)
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
                max_dist, -1, self.num_heads
            )
            edge_input_flat = torch.bmm(
                edge_input_flat,
                self.edge_dis_encoder.weight.reshape(
                    -1, self.num_heads, self.num_heads
                )[:max_dist, :, :],
            )
            edge_input = edge_input_flat.reshape(
                max_dist, n_graph, n_node, n_node, self.num_heads
            ).permute(1, 2, 3, 0, 4)
            edge_input = (
                edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))
            ).permute(0, 3, 1, 2)
        else:
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)


        return edge_input#[n_graph, n_head, n_node, n_node]



