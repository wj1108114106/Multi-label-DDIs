# -*- coding: utf-8 -*-

import os
import gc
import time

import numpy as np
from collections import defaultdict
from keras import backend as K
from keras import optimizers

from utils import load_data, pickle_load, format_filename, write_log
from models import MultiAttention
from models import Multi-label_DDIs
from config import ModelConfig, PROCESSED_DATA_DIR,  ENTITY_VOCAB_TEMPLATE, \
    RELATION_VOCAB_TEMPLATE, ADJ_ENTITY_TEMPLATE, ADJ_RELATION_TEMPLATE, LOG_DIR, PERFORMANCE_LOG, \
    DRUG_VOCAB_TEMPLATE,RELATION_VECTOR_TEMPLATE, B_MATRIX,SMILE_HASH,DRUG_SMILE_TEMPLATE

import matplotlib.pyplot as plt


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_optimizer(op_type, learning_rate):
    if op_type == 'sgd':
        return optimizers.SGD(learning_rate)
    elif op_type == 'rmsprop':
        return optimizers.RMSprop(learning_rate)
    elif op_type == 'adagrad':
        return optimizers.Adagrad(learning_rate)
    elif op_type == 'adadelta':
        return optimizers.Adadelta(learning_rate)
    elif op_type == 'adam':
        return optimizers.Adam(learning_rate, clipnorm=5)
    else:
        raise ValueError('Optimizer Not Understood: {}'.format(op_type))


def train(train_d,dev_d,test_d,kfold,dataset, neighbor_sample_size, embed_dim, n_depth, l2_weight, routings, lr, optimizer_type,
          batch_size, aggregator_type, n_epoch, callbacks_to_add=None, overwrite=True):
    config = ModelConfig()
    config.neighbor_sample_size = neighbor_sample_size
    config.embed_dim = embed_dim
    config.n_depth = n_depth
    config.l2_weight = l2_weight
    config.dataset=dataset
    config.K_Fold=kfold
    config.routings = routings
    config.lr = lr
    config.optimizer = get_optimizer(optimizer_type, lr)
    config.batch_size = batch_size
    config.aggregator_type = aggregator_type
    config.n_epoch = n_epoch
    config.callbacks_to_add = callbacks_to_add

    #drug id
    #should be SMILES
    config.drug_vocab_size = len(pickle_load(format_filename(PROCESSED_DATA_DIR,
                                                             DRUG_VOCAB_TEMPLATE,
                                                             dataset=dataset)))

    #entity id
    config.entity_vocab_size = len(pickle_load(format_filename(PROCESSED_DATA_DIR,
                                                               ENTITY_VOCAB_TEMPLATE,
                                                               dataset=dataset)))

    #relation id
    #string
    config.relation_vocab_size = len(pickle_load(format_filename(PROCESSED_DATA_DIR,
                                                                 RELATION_VOCAB_TEMPLATE,
                                                                 dataset=dataset)))
    #chosen entity matrix
    config.adj_entity = np.load(format_filename(PROCESSED_DATA_DIR, ADJ_ENTITY_TEMPLATE,
                                                dataset=dataset))


    config.adj_relation = np.load(format_filename(PROCESSED_DATA_DIR, ADJ_RELATION_TEMPLATE,
                                                  dataset=dataset))
    ####add 2 row
    config.drug_smile = np.load(format_filename(PROCESSED_DATA_DIR, DRUG_SMILE_TEMPLATE), allow_pickle=True)

    config.smile_hash = np.load(format_filename(PROCESSED_DATA_DIR, SMILE_HASH), allow_pickle=True)


    config.B_matrix = np.load(format_filename(PROCESSED_DATA_DIR, B_MATRIX,
                                                  dataset=dataset))


    config.exp_name = f'Multi-label_DDIs_{dataset}_neigh_{neighbor_sample_size}_embed_{embed_dim}_depth_' \
                      f'{n_depth}_agg_{aggregator_type}_optimizer_{optimizer_type}_lr_{lr}_' \
                      f'batch_size_{batch_size}_epoch_{n_epoch}_routing_{routings}'
    callback_str = '_' + '_'.join(config.callbacks_to_add)
    callback_str = callback_str.replace('_modelcheckpoint', '').replace('_earlystopping', '')#去掉了这两种方式使用swa得方式平均
    config.exp_name += callback_str

    train_log = {'exp_name': config.exp_name, 'batch_size': batch_size, 'optimizer': optimizer_type,
                 'epoch': n_epoch, 'learning_rate': lr}
    print('Logging Info - Experiment: %s' % config.exp_name)
    model_save_path = os.path.join(config.checkpoint_dir, '{}.hdf5'.format(config.exp_name))
    model = Multi-label_DDIs(config)


    train_data=np.array(train_d)
    valid_data=np.array(dev_d)
    test_data=np.array(test_d)
    if not os.path.exists(model_save_path) or overwrite:
        start_time = time.time()
        model.fit(x_train=[train_data[:, :1], train_data[:, 1:2], train_data[:, 2:3]], y_train=train_data[:, 3:4],
                  x_valid=[valid_data[:, :1], valid_data[:, 1:2], valid_data[:, 2:3]], y_valid=valid_data[:, 3:4])
        elapsed_time = time.time() - start_time
        # print('Logging Info - Training time: %s' % time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        train_log['train_time'] = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    # print('Logging Info - Evaluate over valid data:')
    model.load_best_model()
    auc, acc, p, r, f1,aupr, fpr, tpr,precison,recall = model.score(x=[valid_data[:, :1], valid_data[:, 1:2], valid_data[:, 2:3]], y=valid_data[:, 3:4])
    np.save('save_fpr_val_'+dataset+'Fold'+str(kfold),fpr)
    np.save('save_tpr_val_'+dataset+'Fold'+str(kfold),tpr)
    np.save('save_auc_val_'+dataset+'Fold'+str(kfold),auc)

    np.save('save_p_val_' + dataset + 'Fold' + str(kfold), precison)
    np.save('save_r_val_' + dataset + 'Fold' + str(kfold), recall)
    np.save('save_aupr_val_' + dataset + 'Fold' + str(kfold), aupr)

    # lw = 2
    # plt.plot(fpr, tpr,
    #          lw=lw, label='Fold'+str(kfold)+' (area = %0.2f)' % auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # # plt.xticks(font="Times New Roman",size=18,weight="bold")
    # # plt.yticks(font="Times New Roman",size=18,weight="bold")
    # fontsize = 14
    # plt.xlabel('False Positive Rate', fontsize=fontsize)
    # plt.ylabel('True Positive Rate', fontsize=fontsize)
    # plt.title('ROC Curve on'+dataset, fontsize = fontsize)
    # plt.legend(loc="lower right")
    # plt.savefig("auc_val" +str(kfold)+ ".pdf")
    # print(f'Logging Info - dev_auc: {auc}, dev_acc: {acc}, dev_f1: {f1}, dev_aupr: {aupr}')
    train_log['dev_auc'] = auc
    train_log['dev_acc'] = acc
    train_log['dev_p'] = p
    train_log['dev_r'] = r
    train_log['dev_f1'] = f1
    train_log['dev_aupr']=aupr
    train_log['dev_fpr']=fpr
    train_log['dev_tpr']=tpr
    train_log['k_fold']=kfold
    train_log['dataset']=dataset
    train_log['aggregate_type']=config.aggregator_type
    if 'swa' in config.callbacks_to_add:
        model.load_swa_model()
        print('Logging Info - Evaluate over valid data based on swa model:')
        auc, acc, p, r, f1,aupr, fpr, tpr,precison,recall = model.score(x=[valid_data[:, :1], valid_data[:, 1:2], valid_data[:, 2:3]], y=valid_data[:, 3:4])

        train_log['swa_dev_auc'] = auc
        train_log['swa_dev_acc'] = acc
        train_log['swa_dev_p'] = p
        train_log['swa_dev_r'] = r
        train_log['swa_dev_f1'] = f1
        train_log['swa_dev_aupr']=aupr
        print(f'Logging Info - swa_dev_auc: {auc}, swa_dev_acc: {acc}, swa_dev_f1: {f1}, swa_dev_aupr: {aupr}') #修改输出指标
    print('Logging Info - Evaluate over test data:')
    model.load_best_model()
    auc, acc, p, r, f1,aupr, fpr, tpr,precison,recall = model.score(x=[test_data[:, :1], test_data[:, 1:2], test_data[:, 2:3]], y=test_data[:, 3:4])
    np.save('save_fpr_test_' + dataset + 'Fold' + str(kfold), fpr)
    np.save('save_tpr_test_' + dataset + 'Fold' + str(kfold), tpr)
    np.save('save_auc_test_' + dataset + 'Fold' + str(kfold), auc)

    np.save('save_p_test_' + dataset + 'Fold' + str(kfold), precison)
    np.save('save_r_test_' + dataset + 'Fold' + str(kfold), recall)
    np.save('save_aupr_test_' + dataset + 'Fold' + str(kfold), aupr)

    train_log['test_auc'] = auc
    train_log['test_acc'] = acc
    train_log['test_p'] = p
    train_log['test_r'] = r
    train_log['test_f1'] = f1
    train_log['test_aupr'] =aupr
    train_log['dev_fpr'] = fpr
    train_log['dev_tpr'] = tpr
    # print(f'Logging Info - test_auc: {auc}, test_acc: {acc}, test_p: {p}, test_r: {r}, test_f1: {f1}, test_aupr: {aupr}, test_fpr: {fpr}', )
    if 'swa' in config.callbacks_to_add:
        model.load_swa_model()
        print('Logging Info - Evaluate over test data based on swa model:')
        auc, acc, p, r, f1,aupr, fpr, tpr,precison,recall = model.score(x=[test_data[:, :1], test_data[:, 1:2], test_data[:, 2:3]], y=test_data[:, 3:4])
        train_log['swa_test_auc'] = auc
        train_log['swa_test_acc'] = acc
        train_log['swa_test_p'] = p
        train_log['swa_test_r'] = r
        train_log['swa_test_f1'] = f1
        train_log['swa_test_aupr'] = aupr
        print(f'Logging Info - swa_test_auc: {auc}, swa_test_acc: {acc}, swa_test_p: {p}, swa_test_r: {r},swa_test_f1: {f1}, swa_test_aupr: {aupr}')
    train_log['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    write_log(format_filename(LOG_DIR, PERFORMANCE_LOG), log=train_log, mode='a')
    del model
    gc.collect()
    K.clear_session()
    return train_log

