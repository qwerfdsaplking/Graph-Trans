from graphtrasformer.gt_models import *

def get_model(args):
    return GraphTransformer(
        num_encoder_layers = args.num_encoder_layers,
        hidden_dim = args.hidden_dim,
        ffn_hidden_dim = args.ffn_hidden_dim,
        num_attn_heads = args.num_attn_heads,
        emb_dropout= args.emb_dropout,
        dropout = args.dropout,
        attn_dropout = args.attn_dropout,
        num_class = args.num_class,
        encoder_normalize_before = args.encoder_normalize_before,
        apply_graphormer_init = args.apply_graphormer_init,
        activation_fn = args.activation_fn,
        n_trans_layers_to_freeze = args.n_trans_layers_to_freeze,
        traceable = args.traceable,

        use_super_node = args.use_super_node,
        node_feature_type = args.node_feature_type,
        node_feature_dim = args.node_feature_dim,
        node_level_modules = args.node_level_modules,
        attn_level_modules = args.attn_level_modules,
        attn_mask_modules = args.attn_mask_modules,
        num_atoms = args.num_atoms,
        num_in_degree = args.num_in_degree,
        num_out_degree = args.num_out_degree,
        num_edges = args.num_edges,
        eig_pos_dim = args.eig_pos_dim,
        svd_pos_dim = args.svd_pos_dim,
        num_spatial = args.num_spatial,
        num_edge_dis = args.num_edge_dis,
        edge_type = args.edge_type,
        multi_hop_max_dist = args.multi_hop_max_dist,
        num_hop_bias = args.num_hop_bias,

        use_gnn_layers = args.use_gnn_layers,
        gnn_insert_pos = args.gnn_insert_pos,
        num_gnn_layers = args.num_gnn_layers,
        gnn_type = args.gnn_type,
        gnn_dropout = args.gnn_dropout
    )

