# -*- coding: utf-8 -*-

from keras.layers import *
from keras.regularizers import l2
from keras.models import Model
from keras import backend as K  # use computable function
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve,precision_score,recall_score
import sklearn.metrics as m


import layers
from layers import Aggregator
from layers import Caps
from callbacks import KGCNMetric
import tensorflow as tf
from models.base_model import BaseModel
from keras.layers import Input,LSTM
from yoctol_utils.hash import consistent_hash
from keras.layers import Layer
epsilon = 1e-9
import numpy as np
# from sonnet.python.modules import relational_memory



class Multi-label_DDIs(BaseModel):
    def __init__(self, config):
        super(Multi-label_DDIs, self).__init__(config)




    def build(self):
        input_drug_one = Input(
            #drug_one smile
            shape=(1, ), name='input_drug_one', dtype='int64')
        input_drug_two = Input(
            #drug_two_smile
            shape=(1, ), name='input_drug_two', dtype='int64')
        input_drug_relation = Input(
            # real relation
            shape=(1,), name='input_drug_relation', dtype='int64')

        # embedding container
        entity_embedding = Embedding(input_dim=self.config.entity_vocab_size,
                                     output_dim=self.config.embed_dim,
                                     embeddings_initializer='glorot_normal',
                                     embeddings_regularizer=l2(
                                         self.config.l2_weight),
                                     name='entity_embedding')
        # relation are supposed to feet into a encoder-decoder layer
        relation_embedding = Embedding(input_dim=self.config.relation_vocab_size,
                                       output_dim=self.config.embed_dim,
                                       embeddings_initializer='glorot_normal',
                                       embeddings_regularizer=l2(
                                           self.config.l2_weight),
                                       name='relation_embedding')


        ######drug encoder-decoder embedding results
        ####match id to smiles
        ####original has embedding
        ####add code
        ###encoder-decoder
        drug_hash = Lambda(lambda x: self.get_hash(x), name="get_drug_smile_hash")(input_drug_one)  # batch_size,8,64
        drug_one_embedding = Lambda(lambda x: self.drug_embedding(x), name="get_drug_embedding")(drug_hash)

        # drug_one_embedding = entity_embedding(input_drug_one)####ori code

        # get list
        receptive_list_drug_one = Lambda(lambda x: self.get_receptive_field(x),
                                         name='receptive_filed_drug_one')(input_drug_one)
        neineigh_ent_list_drug_one = receptive_list_drug_one[:self.config.n_depth + 1]
        neigh_rel_list_drug_one = receptive_list_drug_one[self.config.n_depth + 1:]

        # get embeded
        neigh_ent_embed_list_drug_one = [entity_embedding(
            neigh_ent) for neigh_ent in neineigh_ent_list_drug_one]
        neigh_rel_embed_list_drug_one = [relation_embedding(
            neigh_rel) for neigh_rel in neigh_rel_list_drug_one]
        neighbor_embedding = Lambda(lambda x: self.get_neighbor_info(x[0], x[1], x[2]),
                                    name='neighbor_embedding_drug_one')

        for depth in range(self.config.n_depth):
            aggregator = Aggregator[self.config.aggregator_type](
                activation='tanh' if depth == self.config.n_depth - 1 else 'relu',
                regularizer=l2(self.config.l2_weight),
                name=f'aggregator_{depth + 1}_drug_one'
            )

            next_neigh_ent_embed_list_drug_one = []
            for hop in range(self.config.n_depth - depth):
                neighbor_embed = neighbor_embedding([drug_one_embedding, neigh_rel_embed_list_drug_one[hop],
                                                     neigh_ent_embed_list_drug_one[hop + 1]])
                next_entity_embed = aggregator(
                    [neigh_ent_embed_list_drug_one[hop], neighbor_embed])
                next_neigh_ent_embed_list_drug_one.append(next_entity_embed)
            neigh_ent_embed_list_drug_one = next_neigh_ent_embed_list_drug_one

        ###for second drug
        ###encoder-decoder layer
        drug_hash = Lambda(lambda x: self.get_hash(x), name='get_drug2_smile_hash')(input_drug_two)  # ?,8,64
        drug_embedding = Lambda(lambda x: self.drug_embedding(x), name="get_drug2_embedding")(drug_hash)

        # drug_embedding = entity_embedding(input_drug_two)###ori code

        # get list
        receptive_list_drug = Lambda(lambda x: self.get_receptive_field(x),
                                     name='receptive_filed_drug_two')(input_drug_two)
        neigh_ent_list_drug = receptive_list_drug[:self.config.n_depth + 1]
        neigh_rel_list_drug = receptive_list_drug[self.config.n_depth + 1:]

        # get embeded
        neigh_ent_embed_list_drug = [entity_embedding(
            neigh_ent) for neigh_ent in neigh_ent_list_drug]
        neigh_rel_embed_list_drug = [relation_embedding(
            neigh_rel) for neigh_rel in neigh_rel_list_drug]
        neighbor_embedding = Lambda(lambda x: self.get_neighbor_info(x[0], x[1], x[2]),
                                    name='neighbor_embedding_drug_two')

        for depth in range(self.config.n_depth):
            aggregator = Aggregator[self.config.aggregator_type](
                activation='tanh' if depth == self.config.n_depth - 1 else 'relu',
                regularizer=l2(self.config.l2_weight),
                name=f'aggregator_{depth + 1}_drug'
            )

            next_neigh_ent_embed_list_drug = []
            for hop in range(self.config.n_depth - depth):
                neighbor_embed = neighbor_embedding([drug_embedding, neigh_rel_embed_list_drug[hop],
                                                     neigh_ent_embed_list_drug[hop + 1]])
                next_entity_embed = aggregator(
                    [neigh_ent_embed_list_drug[hop], neighbor_embed])
                next_neigh_ent_embed_list_drug.append(next_entity_embed)
            neigh_ent_embed_list_drug = next_neigh_ent_embed_list_drug

        ##model
        drug_relation_embedding = relation_embedding(input_drug_relation)

        drug1_squeeze_embed = Lambda(lambda x: K.squeeze(
            x, axis=1))(neigh_ent_embed_list_drug_one[0])
        drug_squeeze_rel_embed= Lambda(lambda x: K.squeeze(
            x, axis=1))(drug_relation_embedding)
        drug2_squeeze_embed = Lambda(lambda x: K.squeeze(
            x, axis=1))(neigh_ent_embed_list_drug[0])
        ########################仿写
        init_states=Concatenate(axis=-1)([drug1_squeeze_embed,drug_squeeze_rel_embed,drug2_squeeze_embed])
        # init_states=Reshape(target_shape=(32,),name='reshape')(drug_relation_embedding)
        init_states=Dense(96,activation='relu',kernel_initializer='he_normal')(init_states)
        init_states=RepeatVector(1)(init_states)

        output1=Lambda(lambda x: self.capsule(x[0],x[1]),name="memory1")([drug1_squeeze_embed,init_states])
        mem_output1 = Lambda(lambda x: x[:, 0:1])(output1)
        memory_input_next_step1=Lambda(lambda x:x[:,1:2])(output1)
        output2=Lambda(lambda x: self.capsule(x[0],x[1]),name="memory2")([drug_squeeze_rel_embed,memory_input_next_step1])
        mem_output2 = Lambda(lambda x: x[:, 0:1])(output2)
        memory_input_next_step2 = Lambda(lambda x: x[:, 1:2])(output2)
        output3=Lambda(lambda x: self.capsule(x[0],x[1]),name="memory3")([drug2_squeeze_embed,memory_input_next_step2])
        mem_output3 = Lambda(lambda x: x[:, 0:1])(output3)
        memory_input_next_step = Lambda(lambda x: x[:, 1:2])(output3)





        ########改写 relational memory没有要mlp 参考本项目下sonnet/python/modules/relational_memory



        #########cnn-maxpool decode
     ######################################
        # mem_output_hr=Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))([mem_output1, mem_output2])
        # mem_output_rt=Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))([mem_output2, mem_output3])
        # mem_output = Lambda(lambda x: K.expand_dims(x, axis=-1))(mem_output_hr)
        # mem_output = Lambda(lambda x: K.expand_dims(x, axis=-1))(mem_output_rt)




        mem_output = Lambda(lambda x: K.concatenate([x[0], x[1], x[2]], axis=1))(
            [mem_output1, mem_output2, mem_output3])
        input_pool=Lambda(lambda x: tf.reshape(x,shape=[-1,288]))(mem_output)
        # mem_output = Lambda(lambda x: K.expand_dims(x, axis=-1))(mem_output)
        # input_conv = Conv2D(64, (3, 1), activation='relu')(mem_output)
        # input_pool=Lambda(lambda x: tf.reshape(tf.nn.top_k(tf.transpose(x,[0,1,3,2]),k=2)[0],shape=[-1,128]))(input_conv)
        input_pool=Dense(100,activation='relu')(input_pool)
        input_pool=Dense(50,activation='relu')(input_pool)

        #######ori maxpooling2d
        # input_pool = MaxPooling2D(pool_size=(1, 96))(input_conv)
        # # input_pool=Conv2D(32,(1,1),activation='relu')(input_pool)
        # input_pool = Lambda(lambda x: K.squeeze(x, axis=1))(input_pool)
        # input_pool = Lambda(lambda x: K.squeeze(x, axis=1))(input_pool)
        # # mem_output = Lambda(lambda x: K.concatenate([x[0], x[1], x[2]], axis=-1))([mem_output1, mem_output2, mem_output3])
        # # input_pool = Lambda(lambda x: K.squeeze(x, axis=1))(mem_output)
        # input_pool=Dense(32,activation='relu')(input_pool)
        ################
        drug_drug_score = Dense(1, activation='sigmoid')(input_pool)


        model = Model([input_drug_one, input_drug_two, input_drug_relation], drug_drug_score)
        model.compile(optimizer=self.config.optimizer,
                      loss='binary_crossentropy', metrics=['acc'])
        return model

    def get_hash(self, drugid):

        ##load smile matrix

        drug_smile_matrix =  K.variable(
            self.config.smile_hash, name='smile_hash')

        drug_smile_entity = K.gather(drug_smile_matrix, K.cast(
                drugid, dtype='int64')) #get drug's hash
        print(drug_smile_entity)

        drug_hash_embed = K.reshape(drug_smile_entity, (-1,int(self.config.timestep),int(self.config.latent_dim/self.config.timestep),))#?,8,64

        return drug_hash_embed

    def drug_embedding(self, input_drug):

        ##encoder-decoder
        drug_embed = input_drug
        print(drug_embed)
        encoder =Bidirectional(LSTM(self.config.latent_dim, return_sequences=True,return_state=True,dropout=0.5))#64->512
        encoder_outputs, state_h, state_c,b_state_h,b_state_c= encoder(drug_embed)
        state_hfinal=Concatenate(axis=-1)([state_h,b_state_h])
        state_cfinal=Concatenate(axis=-1)([state_c,b_state_c])
        encoder_states = [state_hfinal,state_cfinal]
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        # encoder_outputs = K.reshape(encoder_outputs,(-1,1,self.config.latent_dim*2))
        decoder_lstm = LSTM(self.config.latent_dim*2, return_sequences=True, return_state=True,dropout=0.5)
        decoder_outputs, _, _ = decoder_lstm(drug_embed,
                                             initial_state=encoder_states)
        ###attention
        attention=Dense(1,activation='tanh')(encoder_outputs)
        attention=Flatten()(attention)
        attention=Activation('softmax')(attention)
        attention=RepeatVector(self.config.latent_dim*2)(attention)
        attention=Permute([2,1])(attention)
        sent_dense=Multiply()([decoder_outputs,attention])
        decoder_outputs = Dense(self.config.embed_dim, activation='softmax')(sent_dense)
        decoder_outputs=AveragePooling1D(8)(decoder_outputs)# drug_emebdding_end

        print("decoder_outputs")
        print(decoder_outputs)

        return decoder_outputs
    def get_receptive_field(self, entity):
        """Calculate receptive field for entity using adjacent matrix

        :param entity: a tensor shaped [batch_size, 1]
        :return: a list of tensor: [[batch_size, 1], [batch_size, neighbor_sample_size],
                                   [batch_size, neighbor_sample_size**2], ...]
        """
        neigh_ent_list = [entity]
        neigh_rel_list = []
        adj_entity_matrix = K.variable(
            self.config.adj_entity, name='adj_entity', dtype='int64')
        adj_relation_matrix = K.variable(self.config.adj_relation, name='adj_relation',
                                         dtype='int64')
        n_neighbor = K.shape(adj_entity_matrix)[1]

        for i in range(self.config.n_depth):
            new_neigh_ent = K.gather(adj_entity_matrix, K.cast(
                neigh_ent_list[-1], dtype='int64'))  # cast function used to transform data type
            print(new_neigh_ent)
            new_neigh_rel = K.gather(adj_relation_matrix, K.cast(
                neigh_ent_list[-1], dtype='int64'))
            neigh_ent_list.append(
                K.reshape(new_neigh_ent, (-1, n_neighbor ** (i + 1))))
            neigh_rel_list.append(
                K.reshape(new_neigh_rel, (-1, n_neighbor ** (i + 1))))

        return neigh_ent_list + neigh_rel_list

    def get_neighbor_info(self, drug, rel, ent):
        """Get neighbor representation.

        :param user: a tensor shaped [batch_size, 1, embed_dim]
        :param rel: a tensor shaped [batch_size, neighbor_size ** hop, embed_dim]
        :param ent: a tensor shaped [batch_size, neighbor_size ** hop, embed_dim]
        :return: a tensor shaped [batch_size, neighbor_size ** (hop -1), embed_dim]
        """
        # [batch_size, neighbor_size ** hop, 1] drug-entity score
        ####考虑关系对邻居节点的聚合 平均聚合
        #改成gate-attentive aggregator

        # concat_embeds=Lambda(lambda x: K.concatenate([x[0],x[1]],axis=2))([rel,ent])
        # self.dense_1 = Dense(self.config.embed_dim, activation='relu')
        # self.dense_2 = Dense(1)
        # out=self.dense_1(concat_embeds)
        # out = LeakyReLU(alpha=0.2)(out)  # out gcn former change log
        #
        # attn_out = self.dense_2(out)
        # attn_weight=Lambda(lambda x: K.softmax(x))(attn_out)
        # out_attn = Multiply()([out,attn_weight])
        ###gat
        # r_mul_t=Dense(self.config.embed_dim)(ent)
        # r_mul_h=Dense(self.config.embed_dim)(drug)
        # rel_add=K.tanh(Add()([r_mul_h,rel]))
        # rel_add=Lambda(lambda x: tf.expand_dims(x,axis=3))(rel_add)
        # r_mul_t_unsq=Lambda(lambda x: tf.expand_dims(x,axis=2))(r_mul_t)
        # att=Lambda(lambda x: tf.einsum('bhkl,bhlc->bhkc',x[0],x[1]))([r_mul_t_unsq,rel_add])
        # att=K.squeeze(att,axis=-1)
        # weighted_ent=att*ent
        # weighted_ent = K.reshape(weighted_ent,
        #                          (K.shape(weighted_ent)[0], -1,
        #                           self.config.neighbor_sample_size, self.config.embed_dim))
        # neighbor_embed = K.sum(weighted_ent, axis=2)

        #######原始得求weight
        drug_rel_score = K.sum(drug * rel, axis=-1, keepdims=True)
        weighted_ent = drug_rel_score * ent
        weighted_ent = K.reshape(weighted_ent,
                                 (K.shape(weighted_ent)[0], -1,
                                  self.config.neighbor_sample_size, self.config.embed_dim))
        neighbor_embed = K.sum(weighted_ent, axis=2)
        return neighbor_embed


    def capsule(self, drug, relation):

        ##drug: shape(?,?,32)
        ##relation: shape(?,1,32)
        drug=Dense(96)(drug)

        drug = K.reshape(drug, (-1, 1, self.config.embed_dim*3))
        ent = K.concatenate([drug, relation], axis=1)

        caps = Caps(self.config.num_heads, input_bias=0.0, forget_bias=1.0, qkv_size=96)
        capsule_output = caps(ent)

        return capsule_output;
    ####这个是原始得capsuel层得 可以参考这个去封装layer层
    # def capsule(self,drug,relation):
    #
    #     ##drug: shape(?,?,32)
    #     ##relation: shape(?,1,32)
    #
    #     drug = K.reshape(drug,(-1,1,self.config.embed_dim))
    #     ent = K.concatenate([drug,relation],axis=1)
    #
    #     caps = Caps(capsule_dim=32, capsule_num=1, routings=self.config.routings)
    #     capsule_output = caps(ent)
    #     capsule_output = K.reshape(capsule_output,(-1,self.config.embed_dim))
    #
    #     return capsule_output;

    def add_metrics(self, x_train, y_train, x_valid, y_valid):
        self.callbacks.append(KGCNMetric(x_train, y_train, x_valid, y_valid,
                                         self.config.aggregator_type, self.config.dataset, self.config.K_Fold))

    def fit(self, x_train, y_train, x_valid, y_valid):
        self.callbacks = []
        self.add_metrics(x_train, y_train, x_valid, y_valid)
        self.init_callbacks()

        print('Logging Info - Start training...')
        self.model.fit(x=x_train, y=y_train, batch_size=self.config.batch_size,
                       epochs=self.config.n_epoch, validation_data=(
                           x_valid, y_valid), shuffle = True,
                       callbacks=self.callbacks)
        print('Logging Info - training end...')

    def predict(self, x):
        return self.model.predict(x).flatten()

    def score(self, x, y, threshold=0.5):
        y_true = y.flatten()
        y_pred = self.model.predict(x).flatten()
        auc = roc_auc_score(y_true=y_true, y_score=y_pred)
        from sklearn.metrics import roc_curve
        fpr, tpr, thr = roc_curve(y_true=y_true, y_score=y_pred)

        precision, recall, t = precision_recall_curve(y_true=y_true, probas_pred=y_pred)
        aupr = m.auc(recall, precision)
        y_pred = [1 if prob >= threshold else 0 for prob in y_pred]
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        #precision = precision_score()
        p = precision_score(y_true=y_true, y_pred=y_pred)
        r = recall_score(y_true=y_true, y_pred=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred)

        return auc, acc, p,r, f1, aupr, fpr.tolist(), tpr.tolist(),precision.tolist(),recall.tolist()
