'''
@Author: your name
@Date: 2019-12-20 19:02:25
@LastEditTime: 2020-05-26 20:58:12
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /matengfei/KGCN_Keras-master/config.py
'''
# -*- coding: utf-8 -*-

import os

RAW_DATA_DIR = os.getcwd()+'/raw_data'
PROCESSED_DATA_DIR = os.getcwd()+'/data'
LOG_DIR = os.getcwd()+'/log'
MODEL_SAVED_DIR = os.getcwd()+'/ckpt'

KG_FILE = {
           'drugbank':os.path.join(RAW_DATA_DIR,'drugbank','train2id.txt'),
           'kegg':os.path.join(RAW_DATA_DIR,'kegg','train2id.txt'),
    'ogb':os.path.join(RAW_DATA_DIR,'ogb','new_train2id.txt')}
ENTITY2ID_FILE = {
                    'drugbank':os.path.join(RAW_DATA_DIR,'drugbank','entity2id.txt'),
                    'kegg':os.path.join(RAW_DATA_DIR,'kegg','entity2id.txt'),
    'ogb':os.path.join(RAW_DATA_DIR,'ogb','entity2id.txt')}
EXAMPLE_FILE = {
               'drugbank':os.path.join(RAW_DATA_DIR,'drugbank','approved_example2.txt'),
               'kegg':os.path.join(RAW_DATA_DIR,'kegg','approved_example2.txt'),
    'ogb':os.path.join(RAW_DATA_DIR,'ogb','approved_example.txt')}
SEPARATOR = {'drugbank':'\t','kegg':'\t','ogb':' '}
THRESHOLD = {'drugbank':4,'kegg':4,'ogb':4} #添加drug修改
NEIGHBOR_SIZE = {'drugbank':4,'kegg':4,'ogb':4}

#
DRUG_VOCAB_TEMPLATE = '{dataset}_drug_vocab.pkl'
ENTITY_VOCAB_TEMPLATE = '{dataset}_entity_vocab.pkl'
RELATION_VOCAB_TEMPLATE = '{dataset}_relation_vocab.pkl'
ADJ_ENTITY_TEMPLATE = '{dataset}_adj_entity.npy'
ADJ_RELATION_TEMPLATE = '{dataset}_adj_relation.npy'
DRUG_SMILE_TEMPLATE = 'drugbank_id_smile.npy'
SMILE_HASH = 'drugbank_smile_hash.npy'
B_MATRIX = '{dataset}_b_matrix.npy'
RELATION_VECTOR_TEMPLATE = '{dataset}_relation_vector.pkl'
TRAIN_DATA_TEMPLATE = '{dataset}_train.npy'
DEV_DATA_TEMPLATE = '{dataset}_dev.npy'
TEST_DATA_TEMPLATE = '{dataset}_test.npy'
RESULT_LOG={'drugbank':'drugbank_result.txt','kegg':'kegg_result.txt','ogb':'ogb_result.txt'}
PERFORMANCE_LOG = 'KGCNS_performance.log'
DRUG_EXAMPLE='{dataset}_examples.npy'

class ModelConfig(object):
    def __init__(self):
        self.neighbor_sample_size = 16 # neighbor sampling size
        self.embed_dim = 32  # dimension of embedding
        self.n_depth = 2    # depth of receptive field
        self.l2_weight = 1e-7  # l2 regularizer weight
        self.lr = 0.002  # learning rate 0.005
        self.batch_size = 65536
        self.aggregator_type = 'sum'
        self.n_epoch = 50
        self.iters = 1
        self.routings = 1
        self.optimizer = 'adam'

        self.drug_vocab_size = None
        self.entity_vocab_size = None
        self.relation_vocab_size = None
        self.adj_entity = None
        self.adj_relation = None
        self.drug_smile = None
        self.smile_hash = None
        self.relation_vector = None
        self.B_matrix = None

        self.exp_name = None
        self.model_name = None

        # checkpoint configuration 设置检查点
        self.checkpoint_dir = MODEL_SAVED_DIR
        self.checkpoint_monitor = 'val_auc'
        self.checkpoint_save_best_only = True
        self.checkpoint_save_weights_only = True
        self.checkpoint_save_weights_mode = 'max'
        self.checkpoint_verbose = 1

        # early_stoping configuration
        self.early_stopping_monitor = 'val_auc'
        self.early_stopping_mode = 'max'
        self.early_stopping_patience = 5
        self.early_stopping_verbose = 1
        self.dataset='drug'
        self.K_Fold=1
        self.callbacks_to_add = None

        # config for learning rating scheduler and ensembler
        self.swa_start = 3

        ##encoder-decoder
        self.latent_dim = 512

        self.timestep = 8
        self.mem_slots=1
        self.head_size=32
        self.num_heads=3
        self.hidden_size=32
        self.gate_style="memory"
        self.attention_mlp_layers=2
        self.batch_seq_size=16
