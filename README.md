

# Graph-Transformer Framework

Source code for the paper "**[Transformer for Graphs: An Overview from Architecture Perspective](https://arxiv.org/pdf/2202.08455.pdf)**"


We provide a comprehensive review of various Graph Transformer models from the architectural design perspective.
We first disassemble the existing models and conclude three typical ways to incorporate the graph
information into the vanilla Transformer: 
- GNNs as Auxiliary Modules,
- Improved Positional Embedding from Graphs
- Improved Attention Matrix from Graphs. 

We implement the representative components in three groups and conduct a comprehensive comparison on various kinds of famous graph data benchmarks to investigate the real performance gain of each component. 





## Running

- Train and Evaluation. Please see details in our code annotations.
    ```
    $python run.py --seed ${CUSTOMIZED_SEED} \
                   --model_scale ${CUSTOMIZED_SCALE} \
                   --data_name ${CUSTOMIZED_DATASET} \ 
                   --use_super_node ${True/False} \
                   --node_level_modules ${CUSTOMIZED_NODE_MODULES} \ 
                   --attn_level_modules ${CUSTOMIZED_ATTENTION_MODULES} \
                   --attn_mask_modules ${CUSTOMIZED_MASK_MODULES} \
                   --use_gnn_layers ${True/False} \
                   --gnn_insert_pos ${CUSTOMIZED_GNN_POSTION} \
                   --gnn_type ${CUSTOMIZED_GNN} \
                   --sampling_algo ${CUSTOMIZED_SAMPLING_ALGORITHMS}
    ```
- Example 1: Transformer with degree postional embedding, spatial encoding, shortest path edge encoding
    
    ```
    $python run.py --seed 1024 \
               --model_scale small \
               --data_name ZINC \ 
               --use_super_node True \
               --node_level_modules degree \ 
               --attn_level_modules spatial,spe \
    ```
- Example 2: Transformer with 1hop attention mask
    ```
    $python run.py --seed 1024 \
                   --model_scale middle \
                   --data_name flickr \ 
                   --use_super_node True \
                   --node_level_modules eig,svd \ 
                   --attn_mask_modules 1hop \
                   --sampling_algo shadowkhop \
                   --depth 2 \
                   --num_neighbors 10
    ```
- Example 3: Transformer with GIN layers before Transformer layers
    ```
    $python run.py --seed 1024 \
                   --model_scale large \
                   --data_name ZINC \ 
                   --use_super_node True \
                   --use_gnn_layers True \
                   --gnn_insert_pos before \
                   --gnn_type GIN 
    ```




## Requirements
- Python 3.x
- pytorch >=1.5.0
- torch-geometric >=2.0.3
- transformers >= 4.8.2
- tensorflow >= 2.3.1
- scikit-learn >= 0.23.2
- ogb >= 1.3.2

## Results
Please refer to our [paper](https://arxiv.org/pdf/2202.08455.pdf)

## Reference
Please cite the paper whenever our graph transformer is used to produce published results or incorporated into other software:
```
@article{min2022transformer,
  title={Transformer for Graphs: An Overview from Architecture Perspective},
  author={Min, Erxue and Chen, Runfa and Bian, Yatao and Xu, Tingyang and Zhao, Kangfei and Huang, Wenbing and Zhao, Peilin and Huang, Junzhou and Ananiadou, Sophia and Rong, Yu},
  journal={arXiv preprint arXiv:2202.08455},
  year={2022}
}
```


