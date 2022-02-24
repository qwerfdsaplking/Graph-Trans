# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional, Tuple
from graphtrasformer.gnn_layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from graphtrasformer.gt_layers import *
from graphtrasformer.layers import *
logger = logging.getLogger(__name__)


def init_graphormer_params(module):
    """
    Initialize the weights specific to the Graphormer Model.
    """
    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)





class GraphTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int = 12,
        hidden_dim: int = 768,
        ffn_hidden_dim: int = 768*3,
        num_attn_heads: int = 32,
        emb_dropout: float = 0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        num_class: int =2 ,
        encoder_normalize_before: bool = False,
        apply_graphormer_init: bool = False,
        activation_fn: str = "gelu",
        n_trans_layers_to_freeze: int = 0,
        traceable = False,

        use_super_node: bool = True,

        node_feature_type: str = 'cate',
        node_feature_dim: int = None,
        num_atoms: int = None,

        node_level_modules: tuple = ('degree'),
        attn_level_modules: tuple = ('sdp','spatial'),
        attn_mask_modules: str = None,

        num_in_degree: int = None,
        num_out_degree: int = None,
        eig_pos_dim: int = None,
        svd_pos_dim: int = None,

        num_spatial: int = None,
        num_edges: int = None,
        num_edge_dis: int = None,
        edge_type: str = None,
        multi_hop_max_dist: int = None,
        num_hop_bias: int=None,

        use_gnn_layers: bool=False,
        gnn_insert_pos: str='before',
        num_gnn_layers: int=1,
        gnn_type: str='GAT',
        gnn_dropout: float=0.5
    ) -> None:

        super().__init__()
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.hidden_dim= hidden_dim
        self.apply_graphormer_init = apply_graphormer_init
        self.traceable=traceable
        self.use_super_node = use_super_node
        self.use_gnn_layers = use_gnn_layers
        self.gnn_insert_pos = gnn_insert_pos
        self.num_attn_heads=num_attn_heads
        self.attn_mask_modules=attn_mask_modules

        if encoder_normalize_before:
            self.emb_layer_norm = nn.LayerNorm(self.hidden_dim)
        else:
            self.emb_layer_norm = None

        #node feature encoder
        self.node_feature_encoder = NodeFeatureEncoder(feat_type=node_feature_type,
                                                       hidden_dim=hidden_dim,
                                                       n_layers=num_encoder_layers,
                                                       num_atoms=num_atoms,
                                                       feat_dim=node_feature_dim
                                                       )

        if use_super_node:
            self.add_super_node = AddSuperNode(hidden_dim=hidden_dim)

        #node-level graph-structural feature encoder
        self.node_level_layers = nn.ModuleList([])
        for module_name in node_level_modules:
            if module_name=='degree':
                layer = DegreeEncoder(num_in_degree=num_in_degree,
                                      num_out_degree=num_out_degree,
                                      hidden_dim=hidden_dim,
                                      n_layers=num_encoder_layers)
            elif module_name=='eig':
                layer = Eig_Embedding(eig_dim=eig_pos_dim,hidden_dim=hidden_dim)
            elif module_name=='svd':
                layer = SVD_Embedding(svd_dim=svd_pos_dim,hidden_dim=hidden_dim)
            else:
                raise ValueError('node level module error!')
            self.node_level_layers.append(layer)
        #attention-level graph-structural feature encoder
        self.attn_level_layers = nn.ModuleList([])
        for module_name in attn_level_modules:
            if module_name=='spatial':
                layer = GraphAttnSpatialBias(num_heads=num_attn_heads,
                                             num_spatial=num_spatial,
                                             n_layers=num_encoder_layers,
                                             use_super_node=use_super_node)
            elif module_name=='sdp':
                layer = GraphAttnEdgeBias(num_heads = num_attn_heads,
                                          num_edges = num_edges,
                                          num_edge_dis = num_edge_dis,
                                          edge_type=edge_type,
                                          multi_hop_max_dist=multi_hop_max_dist,
                                          n_layers=num_encoder_layers)
            elif module_name=='nhop':
                layer = GraphAttnHopBias(num_heads  = num_attn_heads,
                                         n_hops = num_hop_bias,
                                         use_super_node=use_super_node)
            else:
                raise ValueError('attn level module error!')
            self.attn_level_layers.append(layer)


        #attention mask




        #gnn layers
        if use_gnn_layers:
            if gnn_insert_pos=='before':
                self.gnn_layers = Geometric_GNN(gnn_type=gnn_type,
                                               hidden_dim=hidden_dim,
                                               gnn_dropout=gnn_dropout,
                                               n_layers=num_gnn_layers,
                                                use_super_node=use_super_node)
            elif gnn_insert_pos in ('alter','parallel'):
                self.gnn_layers = nn.ModuleList([Geometric_GNN(gnn_type=gnn_type,
                                               hidden_dim=hidden_dim,
                                               gnn_dropout=gnn_dropout,
                                               n_layers=num_gnn_layers,
                                                use_super_node=use_super_node) for _ in range(num_encoder_layers)])


        #transformer layers
        self.transformer_layers =nn.ModuleList([
            Transformer_Layer(
                num_heads=num_attn_heads,
                hidden_dim=hidden_dim,
                ffn_hidden_dim=ffn_hidden_dim,
                dropout=dropout,
                attn_dropout=attn_dropout,
                temperature=1,
                activation_fn=activation_fn
            ) for _ in range(num_encoder_layers)
        ])


        self.output_layer_norm = nn.LayerNorm(hidden_dim)
        self.output_fc1 = nn.Linear(hidden_dim,hidden_dim)
        self.output_fc2 = nn.Linear(hidden_dim,num_class)
        self.out_act_fn = get_activation_function(activation_fn)


        # Apply initialization of model params after building the model
        if self.apply_graphormer_init:
            self.apply(init_graphormer_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])



    def forward(
        self,
        batched_data,
        perturb=None,
        last_state_only: bool = False,
    ):

        #==============preparation==========================
        # compute padding mask. This is needed for multi-head attention
        data_x = batched_data["x"]
        n_graph, n_node = data_x.size()[:2]

        #calculate attention padding mask # B x T x T / Bx T+1 xT+1
        padding_mask = batched_data['x_mask']
        if self.use_super_node:
            padding_mask_cls = torch.ones(
                n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype
            )
            padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1).float()
        attn_mask = torch.matmul(padding_mask.unsqueeze(-1), padding_mask.unsqueeze(1)).long()
        self.attn_mask=attn_mask

        #x feature encode
        x = self.node_feature_encoder(batched_data)# B x T x C
        for nl_layer in self.node_level_layers:
            node_bias = nl_layer(batched_data)
            x += node_bias
        #add the super node
        if self.use_super_node:
            x = self.add_super_node(x)# B x T+1 x C
        #perturbation
        if perturb is not None:
            #x[:, 1:, :] += perturb
            pass

        # attention bias computation,  B x H x (T+1) x (T+1)  or B x H x T x T
        attn_bias = torch.zeros(n_graph,self.num_attn_heads,n_node+int(self.use_super_node),n_node+int(self.use_super_node)).to(data_x.device)
        for al_layer in self.attn_level_layers:
            bias = al_layer(batched_data)
            if bias.shape[-1]==attn_bias.shape[-1]:
                attn_bias+=bias
            elif bias.shape[-1]==attn_bias.shape[-1]-1:
                attn_bias[:, :, int(self.use_super_node):, int(self.use_super_node):] = attn_bias[:, :, int(self.use_super_node):, int(self.use_super_node):] + bias
            else:
                raise ValueError('attention calculation error')

        #attention mask TODO
        if self.attn_mask_modules in ('1hop','nhop'):
            adj_mask = getAttnMasks(batched_data,self.attn_mask_modules,self.use_super_node,self.num_attn_heads)
            attn_mask = attn_mask.unsqueeze(1).expand(-1,self.num_attn_heads,-1,-1)*adj_mask
        #===================data flow===============
        #input feature normalization and dropout
        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)
        x = self.emb_dropout(x)   # B x T+1 x C

        #gnn layers before transformer
        if self.use_gnn_layers and self.gnn_insert_pos=='before':
            x = self.gnn_layers(batched_data,x)


        # graph transformer layers
        inner_states = []
        if not last_state_only:
            inner_states.append(x)
        for i,layer in enumerate(self.transformer_layers):

            if self.use_gnn_layers and self.gnn_insert_pos=='parallel':
                x_graph = self.gnn_layers[i](batched_data, x)
            else:
                x_graph = 0

            #self-attention layer
            x, _ = layer.attention(
                x=x,
                mask=attn_mask,
                attn_bias=attn_bias,
            )

            if self.use_gnn_layers and self.gnn_insert_pos=='alter':#by default, gnn after mhsa
                x = self.gnn_layers[i](batched_data, x)

            x = x + x_graph


            #FFN layer
            x = layer.ffn_layer(x)

            if not last_state_only:
                inner_states.append(x)



        #output layers
        if self.use_super_node:
            graph_rep = x[:, 0, :].squeeze()#B x 1 x C
        else:
            #center node
            root_n_id =  batched_data['root_n_id']
            root_idx = (torch.arange(n_graph,device=x.device)*n_node+root_n_id).long()
            graph_rep = x.reshape(-1,x.shape[-1])[root_idx].squeeze()
            #mean pooling, other readout methods to be implemented, e.g, center node
            #x = x.reshape(-1, self.hidden_dim)
            #padding_mask = padding_mask.reshape(-1).bool()
            #x[~padding_mask]=0
            #ns = batched_data['ns']#node number in each graph
            #graph_rep = x.reshape(-1,n_node,self.hidden_dim).sum(1) / ns.unsqueeze(1)

        #output transformation
        out = self.output_layer_norm(self.out_act_fn(self.output_fc1(graph_rep)))
        out = self.output_fc2(out).squeeze()

        return {'logits':out}


