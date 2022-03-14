from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from sklearn.metrics import roc_auc_score, average_precision_score
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from argparse import ArgumentParser, Namespace
from data.collator import *
from gt_dataset import *

import gc
from graphtrasformer.architectures import *
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


from sklearn import metrics
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import time
import torch.onnx
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
import warnings
warnings.filterwarnings('ignore')




def str_tuple(string):
    return tuple(string.split(','))

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def set_model_scale(model_scale,args):
    if model_scale=='mini':
        args.num_encoder_layers  = 3
        args.hidden_dim = 64
        args.ffn_hidden_dim = 64
        args.num_attn_heads = 4
    elif model_scale=='small':
        args.num_encoder_layers  = 6
        args.hidden_dim = 80
        args.ffn_hidden_dim = 80
        args.num_attn_heads = 8

    elif model_scale=='middle':
        args.num_encoder_layers  = 12
        args.hidden_dim = 80
        args.ffn_hidden_dim = 80
        args.num_attn_heads = 8
    elif model_scale=='large':
        args.num_encoder_layers  = 12
        args.hidden_dim = 512
        args.ffn_hidden_dim = 512
        args.num_attn_heads = 32
    return args

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--disable_tqdm',type=boolean_string,default=True)#just for debug

    parser.add_argument('--model_scale', type=str, default='small')#('small','middle','large')
    parser.add_argument('--data_name',type=str,default='ZINC')
    parser.add_argument('--data_param',type=str,default=None)
    #basic Transformer parameters
    parser.add_argument('--max_node', type=int, default=512)
    parser.add_argument('--num_encoder_layers', type=int, default=12)
    parser.add_argument('--hidden_dim',         type=int, default=768)
    parser.add_argument('--ffn_hidden_dim',     type=int, default=768*3)
    parser.add_argument('--num_attn_heads',     type=int, default=32)
    parser.add_argument('--emb_dropout',type=float,default=0.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--attn_dropout', type=float, default=0.1)
    parser.add_argument('--num_class', type=int, default=1)
    parser.add_argument('--encoder_normalize_before', type=boolean_string, default=True)
    parser.add_argument('--apply_graphormer_init', type=boolean_string, default=True)
    parser.add_argument('--activation_fn', type=str, default='GELU')
    parser.add_argument('--n_trans_layers_to_freeze', type=int, default=0)
    parser.add_argument('--traceable', type=boolean_string, default=False)



    #various positional embedding parameters
    parser.add_argument('--use_super_node', type=boolean_string, default=True)
    parser.add_argument('--node_feature_type', type=str, default=None)#or dense
    parser.add_argument('--node_feature_dim', type=int, default=None)# valid only for dense feature type
    parser.add_argument('--num_atoms', type=int, default=512*9)# valid only for cate feature type
    parser.add_argument('--node_level_modules', type=str_tuple, default=())#,'eig','svd'))'degree'
    parser.add_argument('--eig_pos_dim',type=int, default=3)#2
    parser.add_argument('--svd_pos_dim',type=int, default=3)
    parser.add_argument('--num_in_degree', type=int, default=512)
    parser.add_argument('--num_out_degree', type=int, default=512)

    #various attention bias/mask parameters
    parser.add_argument('--attn_level_modules', type=str_tuple, default=())#,'nhop'))'spatial',spe
    parser.add_argument('--attn_mask_modules',type=str, default=None)#'nhop'
    parser.add_argument('--num_edges', type=int, default=512*3)
    parser.add_argument('--num_spatial', type=int, default=512)
    parser.add_argument('--num_edge_dis', type=int, default=128)
    parser.add_argument('--spatial_pos_max', type=int, default=20)
    parser.add_argument('--edge_type', type=str, default=None)
    parser.add_argument('--multi_hop_max_dist', type=int, default=5)
    parser.add_argument('--num_hop_bias', type=int, default=3)#2/3/4

    #gnn layers parameters.     Insert gnn layers before/alternate/parallel self-attention layers
    #gnn layers are implemented by pytorch geometric for simplicity, so we always require data transformation across gnn layer and self-attention layers
    parser.add_argument('--use_gnn_layers', type=boolean_string, default=False)
    parser.add_argument('--gnn_insert_pos', type=str, default='before')#'before'/'alter'/'parallel' gnn insert position
    parser.add_argument('--num_gnn_layers', type=int, default=1) #
    parser.add_argument('--gnn_type',type=str,default='GAT') #GCN,SAGE,GAT,RGCN ... any types of GNN supported by Geometric
    parser.add_argument('--gnn_dropout',type=float,default=0.5)


    #sampling parameters
    parser.add_argument('--depth',type=int,default=2)
    parser.add_argument('--num_neighbors', type=int,default=10)
    parser.add_argument('--sampling_algo',type=str,default='shadowkhop')# or sage


    # training parameters, we use Trainer class from Huggingface Transformer, which is highly optimized specifically for Transformer architecture
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output_dir',type=str)#G#v2'./output'
    parser.add_argument('--per_device_train_batch_size', type=int, default=256)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=256)
    parser.add_argument('--gradient_accumulation_steps',type=int,default=1)
    parser.add_argument('--learning_rate',type=float, default=2e-4)
    parser.add_argument('--weight_decay',type=float,default=0.01)
    parser.add_argument('--adam_beta1',type=float,default='0.9')
    parser.add_argument('--adam_beta2',type=float,default='0.999')
    parser.add_argument('--adam_epsilon',type=float,default=1e-8)
    parser.add_argument('--max_grad_norm',type=float,default=5.0)
    parser.add_argument('--num_train_epochs',type=int,default=300)
    parser.add_argument('--max_steps',type=int,default=400000)#1000000
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')
    parser.add_argument('--warmup_steps',type=int,default=40000)
    parser.add_argument('--dataloader_num_workers',type=int,default=32)
    parser.add_argument('--evaluation_strategy',type=str,default='steps')
    parser.add_argument('--eval_steps',type=int,default=1000)
    parser.add_argument('--save_steps',type=int,default=1000)
    parser.add_argument('--greater_is_better',type=boolean_string,default=True)

    parser.add_argument('--rerun',type=boolean_string,default=False)

    args = parser.parse_args()
    set_model_scale(args.model_scale,args)


    return args







def expand_graph_level_dataset(dataset,N):
    train_data = list(dataset.dataset)
    dataset.dataset = N*train_data
    dataset.num_data*=N
    return dataset
def expand_node_level_dataset(dataset,N):
    dataset.node_idx = dataset.node_idx.expand(N,-1).reshape(-1)
    dataset.num_data*=N
    return dataset





if __name__=='__main__':
    args = parse_args()


    #for efficiency in Trainer.
    # I don't know why the Trainer Class will suspend several seconds after each epoch of dataloader,
    # So I expand the training set manually for efficiency
    expand_num_dict={'flickr':20,
                     'ZINC':20,
                     'ogbn-products':5,
                     'ogbg-molpcba':1,
                     'ZINC-full':1,
                     'ogbn-arxiv':1,
                     "ogbg-molhiv":5}


    if args.data_name in ('flickr','ogbn-products','ogbn-arxiv'):
        train_set, valid_set, test_set, odata, args = get_node_level_dataset(args.data_name, args=args)
        train_set = expand_node_level_dataset(train_set,expand_num_dict[args.data_name])

    elif args.data_name in ('ZINC','UPFD','ogbg-molpcba','ZINC-full',"ogbg-molhiv"):
        train_set,valid_set,test_set,odata = get_graph_level_dataset(args.data_name,param=args.data_param,set_default_params=True,args=args)
        train_set = expand_graph_level_dataset(train_set, expand_num_dict[args.data_name])

    else:
        raise ValueError('no dataset')


    criterion, metric, task_type,metric_name=get_loss_and_metric(args.data_name)

    #print parameters
    for k,v in vars(args).items():
        print(k,v)
    #========================model===============================
    model=get_model(args)



    log_file_param_list = (args.data_name,
                           args.model_scale,
                           args.use_super_node,
                           args.node_level_modules,
                           args.eig_pos_dim,
                           args.svd_pos_dim,
                           args.attn_level_modules,
                           args.attn_mask_modules,
                           args.num_hop_bias,
                           args.use_gnn_layers,
                           args.gnn_insert_pos,
                           args.num_gnn_layers,
                           args.gnn_type,
                           args.gnn_dropout,
                           args.sampling_algo,
                           args.depth,
                           args.num_neighbors,
                           args.seed)

    log_file_param_list_p=[]
    for x in log_file_param_list:
        if isinstance(x, tuple):
            if len(x)==0:
                x='None'
            else:
                x='+'.join(x)
        else:
            x=str(x)

        log_file_param_list_p.append(x)

    output_dir ='./outputs/'+'_'.join(log_file_param_list_p)
    log_file_path = output_dir+'/logs.json'

    setattr(args,'log_file_path',log_file_path)
    setattr(args,'output_dir', output_dir)



    ##huggingface trainer============================
    def compute_metrics(p: EvalPrediction):
        preds,labels = p

        gc.collect()
        if task_type=='multi_classification':
            preds = np.argmax(preds, axis=1)
            labels = labels.astype(np.long)
            return metric.compute(predictions=preds, references=labels)

        elif task_type=='multi_binary_classification' and metric_name=='AP':
            preds = torch.sigmoid(torch.tensor(preds)).numpy()
            return {metric_name:metric.eval({'y_true':labels,'y_pred':preds})['ap']}#输入的格式，输出的格式都要确认 #确认node edge特征是否正确处理

        elif task_type=='regression':
            return {metric_name:metric(torch.tensor(preds),torch.tensor(labels)).item()}#mae

        elif task_type=='binary_classification' and metric_name=='ROC-AUC':
            return {metric_name:roc_auc_score(y_true=labels,y_score=torch.sigmoid(torch.tensor(preds)).numpy())}
        elif task_type=='binary_classification' and metric_name=='accuracy':
            return metric.compute(predictions=torch.sigmoid(torch.tensor(preds)), references=labels)



    from transformers import TrainerCallback,TrainerState,TrainerControl,EarlyStoppingCallback
    class MyCallback(TrainerCallback):
        def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            print("save logs...")
            state.save_to_json(log_file_path)



    class MyTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            self.inputs=inputs
            labels = inputs['labels']
            outputs = model(inputs)

            labels = labels.long() if task_type=='multi_classification' else labels.float()
            if task_type=='multi_binary_classification':
                labels = labels.reshape(-1)
                mask = ~torch.isnan(labels)
                loss = criterion(outputs['logits'].reshape(-1)[mask],labels[mask])

            else:
                loss = criterion(outputs['logits'],labels)
            return (loss,outputs) if return_outputs else loss


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        dataloader_num_workers=args.dataloader_num_workers,# sensitive
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        greater_is_better=args.greater_is_better,
        save_steps=args.save_steps,
        save_total_limit=10,
        logging_steps=args.eval_steps,
        seed=args.seed

    )


    training_args.disable_tqdm=args.disable_tqdm
    training_args.ignore_data_skip=True



    resume_from_checkpoint = True if (check_checkpoints(args.output_dir) and not args.rerun) else None

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        compute_metrics=compute_metrics,
        data_collator=lambda x:collator(x,args),
        callbacks=[MyCallback,EarlyStoppingCallback(early_stopping_patience=20)]
    )
    trainer.args._n_gpu=1

    print(trainer.evaluate())
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


    predictions, labels, test_metrics = trainer.predict(test_set, metric_key_prefix="predict")
    test_metrics['best_val_metric']=trainer.state.best_metric
    test_metrics['best_model_checkpoint']=trainer.state.best_model_checkpoint
    f=open(args.output_dir+'/test.txt','w')
    for k,v in test_metrics.items():
        f.write(str(k)+':'+str(v)+'\n')
    f.write('\n')
    f.close()






