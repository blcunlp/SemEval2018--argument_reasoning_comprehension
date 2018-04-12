# coding: utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os
import utils
import math
from six.moves import xrange
from tensorflow.python.util import nest
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.rnn import LSTMCell, BasicLSTMCell
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import DropoutWrapper, ResidualWrapper
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
from utils import print_params

class InitModel:
    def __init__(self, flags,word_index_to_embeddings_map,train_or_others):
        self.lstm_size = flags.lstm_size
        self.max_grad_norm=flags.max_grad_norm
        self.batch_size = flags.batch_size
        self.learning_rate = flags.learning_rate
        self.max_len = flags.max_len
        self.embedding_size = flags.embedding_size
        self.word_index_to_embeddings_map=word_index_to_embeddings_map
        self.num_layers=flags.num_layers
        self.train_or_others = train_or_others
        self.rich_context=flags.rich_context
        self.l2_strength=flags.l2_strength
        self.keep_prob_placeholder = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False)
        print_params(flags)

        self.sentences_placeholder()
        self.get_embedding()
        #sentence to lstm and then ,concat and max pooling
        self.build_sentences()
        #attention for both w0 and w1
        self.attention()
        self.pred_loss()
        self.saver = tf.train.Saver()


    def build_single_cell(self):
        cell = LSTMCell(self.lstm_size)
        if self.train_or_others:
            cell = DropoutWrapper(cell, input_keep_prob=self.keep_prob_placeholder, output_keep_prob=self.keep_prob_placeholder)
        return cell

    def build_multi_cell(self):
        return MultiRNNCell([self.build_single_cell() for i in range(self.num_layers)])

    def sentences_placeholder(self):
        with tf.variable_scope('build_sentences'):
            self.input_length = []
            self.inputs_w0 = tf.placeholder(tf.int32, shape=(None, None), name='inputs_w0')
            self.inputs_w1 = tf.placeholder(tf.int32, shape=(None, None), name='inputs_w1')
            self.inputs_r = tf.placeholder(tf.int32, shape=(None, None), name='inputs_r')
            self.inputs_c = tf.placeholder(tf.int32, shape=(None, None), name='inputs_c')
            self.inputs_d= tf.placeholder(tf.int32, shape=(None, None), name='inputs_d')
            self.input_label=tf.placeholder(tf.int32, shape=(None,), name='input_label')
            w0_inputs_length = tf.placeholder(dtype=tf.int32, shape=(None,), name='w0_inputs_length')
            w1_inputs_length = tf.placeholder(dtype=tf.int32, shape=(None,), name='w1_inputs_length')
            r_inputs_length = tf.placeholder(dtype=tf.int32, shape=(None,), name='r_inputs_length')
            c_inputs_length = tf.placeholder(dtype=tf.int32, shape=(None,), name='c_inputs_length')
            d_inputs_length = tf.placeholder(dtype=tf.int32, shape=(None,), name='d_inputs_length')
            self.input_length = [w0_inputs_length, w1_inputs_length, r_inputs_length, c_inputs_length,d_inputs_length]

    def get_embedding(self):
        # Initialize encoder_embeddings to have variance=1.
        with tf.device("/cpu:0"):
            self.input_emb = []
            embedding_matrix =self.word_index_to_embeddings_map

            embedding = tf.Variable(embedding_matrix, trainable=False, name="embedding")
            input_w0 = tf.nn.embedding_lookup(embedding, self.inputs_w0)
            input_w1 = tf.nn.embedding_lookup(embedding, self.inputs_w1)
            input_reason = tf.nn.embedding_lookup(embedding, self.inputs_r)
            input_claim = tf.nn.embedding_lookup(embedding, self.inputs_c)
            input_debate = tf.nn.embedding_lookup(embedding, self.inputs_d)
            self.input_emb = [input_w0, input_w1, input_reason, input_claim, input_debate]

    def build_sentences(self):
        with tf.variable_scope("generate_lstm") as scope:
            self.enco_inputs_list = []
            self.biLSTM_outputs_list = []
            self.last_sent_represent_list = []
            self.cell_fw = self.build_multi_cell()
            self.cell_bw = self.build_multi_cell()

            for i in xrange(5):
                if i == 0:
                    self.outputs, self.final_state = tf.nn.bidirectional_dynamic_rnn(
                            self.cell_fw, self.cell_bw, self.input_emb[i], sequence_length=self.input_length[i],
                            dtype=tf.float32)

                else:
                    scope.reuse_variables()
                    self.outputs, self.final_state = tf.nn.bidirectional_dynamic_rnn(
                            self.cell_fw, self.cell_bw, self.input_emb[i], sequence_length=self.input_length[i],
                            dtype=tf.float32)
                biLstm_outputs = tf.concat(self.outputs ,2)
                self.biLSTM_outputs_list.append(biLstm_outputs)
                final_state_fw = self.final_state[0]
                final_state_bw = self.final_state[1]
                h_fw = final_state_fw[-1][1]
                h_bw = final_state_bw[-1][1]
                last_sent_represent = tf.concat([h_fw, h_bw], -1)
                self.last_sent_represent_list.append(last_sent_represent)


        if self.rich_context:
            attention_vector_for_w0 = tf.concat([self.last_sent_represent_list[0],self.last_sent_represent_list[2], self.last_sent_represent_list[3], self.last_sent_represent_list[4]], axis=1)
            #q for w0
            self.w_r_0 = tf.get_variable("w_r_0", shape=[8 * self.lstm_size, self.lstm_size],
                                        regularizer=tf.contrib.layers.l2_regularizer(self.l2_strength))
            attention_w0 = tf.matmul(attention_vector_for_w0,self.w_r_0, name="w0_r")
            self.attention_vector_for_w0 = tf.nn.dropout(attention_w0, self.keep_prob_placeholder)
            print("self.attention_vector_for_w0 ######",self.attention_vector_for_w0)
            # q for w1  concat then reduce dim
            attention_vector_for_w1 = tf.concat([self.last_sent_represent_list[1],self.last_sent_represent_list[2], self.last_sent_represent_list[3], self.last_sent_represent_list[4]], axis=1)
            self.w_r_1 = tf.get_variable("w_r_1", shape=[8 * self.lstm_size, self.lstm_size],
                                        regularizer=tf.contrib.layers.l2_regularizer(self.l2_strength))
            attention_w1 = tf.matmul(attention_vector_for_w1,self.w_r_1, name="w1_r")
            self.attention_vector_for_w1 = tf.nn.dropout(attention_w1, self.keep_prob_placeholder)
            print("attention_vector_for_w1*********", self.attention_vector_for_w1)
        else:
            attention_vector_for_w0 = tf.concat([self.last_sent_represent_list[1],self.last_sent_represent_list[2], self.last_sent_represent_list[3]], axis=1)

            #q for w0
            self.w_p_0 = tf.get_variable("w_p_0", shape=[6 * self.lstm_size, self.lstm_size],
                                        regularizer=tf.contrib.layers.l2_regularizer(self.l2_strength))
            attention_w0 = tf.matmul(attention_vector_for_w0,self.w_p_0, name="w0_p")
            self.attention_vector_for_w0 = tf.nn.dropout(attention_w0, self.keep_prob_placeholder)
            print("self.attention_vector_for_w0 ######",self.attention_vector_for_w0)
            # q for w1  concat then reduce dim
            attention_vector_for_w1 = tf.concat([self.last_sent_represent_list[0],self.last_sent_represent_list[2], self.last_sent_represent_list[3]], axis=1)

            self.w_r_1 = tf.get_variable("w_p_1", shape=[6 * self.lstm_size, self.lstm_size],
                                        regularizer=tf.contrib.layers.l2_regularizer(self.l2_strength))
            attention_w1 = tf.matmul(attention_vector_for_w1,self.w_r_1, name="w1_p")
            self.attention_vector_for_w1 = tf.nn.dropout(attention_w1, self.keep_prob_placeholder)
            print("attention_vector_for_w1*********", self.attention_vector_for_w1)

        with tf.variable_scope("encode_x"):
            self.w0_output=self.two_layer_tensordot(self.input_emb[0],self.lstm_size,scope="x_fnn")
            #self.x_output=self.x_output*self.x_mask[:,:,None]

        with tf.variable_scope("encode_y"):
            self.w1_output=self.two_layer_tensordot(self.input_emb[1],self.lstm_size,scope="y_fnn")
            #self.y_output=self.y_output*self.y_mask[:,:,None]

        with tf.variable_scope("dot-product-atten"):
            #weightd_w1:(b,x_len,2*h),weighted_w0:(b,y_len,2*h)
            weighted_w1, weighted_w0 =self.dot_product_attention(x_sen=self.w0_output,y_sen=self.w1_output,x_len = self.max_len,y_len=self.max_len)
            #self.mul_w0 = tf.matmul(weighted_w0,self.biLSTM_outputs_list[0])
            #self.mul_w1 = tf.matmul(weighted_w1,self.biLSTM_outputs_list[1])
            #self.diff_w0 = tf.subtract(self.biLSTM_outputs_list[0],weighted_w0)
            #self.diff_w1 = tf.subtract(self.biLSTM_outputs_list[1],weighted_w1)
    def tensordot(self,inp,out_dim,in_dim=None,activation=None,use_bias=False,w_name="batch-fnn-W"):
        '''
        function: the implement of FNN ,input is a 3D batch tesor,W is a 2D tensor
        :param input: a 3D tensor of (b,seq_len,h)
        :param out_dim: the out_dim of W
        :param in_dim: the in_dim of W
        :param activation: activation function
        :param use_bias: use bias or not
        :param w_name: the unique name for W
        :return: (b,seq_len,in_dim)*(in_dim,out_dim) ->(b,seq_len,out_dim)
        '''
        with tf.variable_scope("3D-batch-fnn-layer"):
            inp_shape = inp.get_shape().as_list()
            batch_size= inp_shape[0]
            seq_len = inp_shape[1]

            if in_dim==None:
                in_dim = inp_shape[-1]

            W = tf.get_variable(w_name,shape=[in_dim,out_dim])
            out = tf.tensordot(inp,W,axes=1)

            if use_bias == True:
                b_name = w_name + '-b'
                b = tf.get_variable(b_name, shape=[out_dim])
                out = out + b

            if activation is not None:
                out = activation(out)
            out.set_shape([batch_size,seq_len,out_dim])
            return out

    def two_layer_tensordot(self,inp,out_dim,scope):
        with tf.variable_scope(scope):
            output1=self.tensordot(inp,out_dim,activation=tf.nn.relu,use_bias=True,w_name="first_layer")
            output2=self.tensordot(output1,out_dim,activation=tf.nn.relu,use_bias=True,w_name="second_layer")
            return output2

    def dot_product_attention(self,x_sen,y_sen,x_len,y_len):
        '''
        function: use the dot-production of left_sen and right_sen to compute the attention weight matrix
        :param left_sen: a list of 2D tensor (x_len,hidden_units)
        :param right_sen: a list of 2D tensor (y_len,hidden_units)
        :return: (1) weighted_y: the weightd sum of y_sen, a 3D tensor with shape (b,x_len,2*h)
               (2)weghted_x:  the weighted sum of x_sen, a 3D tensor with shape (b,y_len,2*h)
        '''
        weight_matrix =tf.matmul(x_sen, tf.transpose(y_sen,perm=[0,2,1])) #(b,x_len,h) x (b,h,y_len)->(b,x_len,y_len)
        #reduce_max or reduce_min***
        weight_matrix_y =tf.exp(weight_matrix - tf.reduce_min(weight_matrix,axis=2,keep_dims=True))  #(b,x_len,y_len)
        weight_matrix_x =tf.exp(tf.transpose((weight_matrix - tf.reduce_min(weight_matrix,axis=1,keep_dims=True)),perm=[0,2,1]))  #(b,y_len,x_len)

        #weight_matrix_y=weight_matrix_y*self.y_mask[:,None,:]#(b,x_len,y_len)*(b,1,y_len)
        #weight_matrix_x=weight_matrix_x*self.x_mask[:,None,:]#(b,y_len,x_len)*(b,1,x_len)

        alpha=weight_matrix_y/(tf.reduce_sum(weight_matrix_y,2,keep_dims=True)+1e-8)#(b,x_len,y_len)
        beta=weight_matrix_x/(tf.reduce_sum(weight_matrix_x,2,keep_dims=True)+1e-8)#(b,y_len,x_len)

        #(b,1,y_len,2*h)*(b,x_len,y_len,1)*=>(b,x_len,y_len,2*h) =>(b,x_len,2*h)
        weighted_y =tf.reduce_sum(tf.expand_dims(y_sen,1) *tf.expand_dims(alpha,-1),2)

        #(b,1,x_len,2*h)*(b,y_len,x_len,1) =>(b,y_len,x_len,2*h) =>(b,y_len,2*h)
        weighted_x =tf.reduce_sum(tf.expand_dims(x_sen,1) * tf.expand_dims(beta,-1),2)

        return weighted_y,weighted_x

    def attention(self):
        batch_size = self.batch_size
        with tf.variable_scope("attetnion"):
            self.w_am = tf.get_variable("w_am", shape=[2*self.lstm_size, self.lstm_size],regularizer=tf.contrib.layers.l2_regularizer(self.l2_strength))
            #warrant0_biLstm_outputs shape is (bat,max_len,lstm_size*2)
            w_am_w0 = tf.matmul(tf.reshape(self.biLSTM_outputs_list[0], shape=[-1, 2*self.lstm_size]),self.w_am, name="wam_w0")
            #w_am_w0 = tf.matmul(self.warrant0_biLstm_outputs,self.w_am, name="wam_w0")
            w_am_w0 = tf.reshape(w_am_w0, shape=[-1, self.max_len, self.lstm_size])
            self.w_qm = tf.get_variable("w_qm", shape=[ self.lstm_size, self.lstm_size],regularizer=tf.contrib.layers.l2_regularizer(self.l2_strength))
            w_qm_w0 = tf.reshape(tf.matmul(self.attention_vector_for_w0,self.w_qm, name="wam_w0"),shape=[-1,1,self.lstm_size])
            m_w0=tf.tanh(w_am_w0+w_qm_w0)

            self.WT = tf.get_variable("WT", shape=[self.lstm_size, 1],regularizer=tf.contrib.layers.l2_regularizer(self.l2_strength))  # h x 1
            WTm_w0 = tf.matmul(tf.reshape(m_w0, shape=[-1, self.lstm_size]), self.WT)
            s_w0 = tf.nn.softmax(
                tf.reshape(WTm_w0, shape=[-1, 1, self.max_len], name="s_w0"))  # b x 1 x max_len
            w0_after_attention = tf.matmul(s_w0, self.biLSTM_outputs_list[0])  # [b x 1 x max_len]*[b x max_len x 2*hidden_units] =[b x max_len  x 2*hidden_units]
            self.w0_after_attention = tf.nn.dropout(w0_after_attention, self.keep_prob_placeholder)


        with tf.variable_scope("w1_attention"):
            self.w_am1 = tf.get_variable("w_am1", shape=[2 * self.lstm_size, self.lstm_size],regularizer=tf.contrib.layers.l2_regularizer(self.l2_strength))
            w_am_w1 = tf.matmul(tf.reshape(self.biLSTM_outputs_list[1], shape=[-1, 2 * self.lstm_size]), self.w_am1, name="wam_w1")
            w_am_w1 = tf.reshape(w_am_w1, shape=[-1, self.max_len, self.lstm_size])
            print("self.w_am_w1", w_am_w1)

            self.w_qm1 = tf.get_variable("w_qm1", shape=[self.lstm_size, self.lstm_size],regularizer=tf.contrib.layers.l2_regularizer(self.l2_strength))
            w_qm_w1 = tf.reshape(tf.matmul(self.attention_vector_for_w1, self.w_qm1, name="wam_w1"),shape=[-1,1,self.lstm_size])
            print("self.w_qm_w1", w_qm_w1)
            m_w1 = tf.tanh(w_am_w1 + w_qm_w1)

            self.WT1 = tf.get_variable("WT1", shape=[self.lstm_size, 1],regularizer=tf.contrib.layers.l2_regularizer(self.l2_strength))  # h x 1
            WTm_w1 = tf.matmul(
                tf.reshape(m_w1, shape=[-1, self.lstm_size]),
                self.WT1)
            s_w1 = tf.nn.softmax(
                tf.reshape(WTm_w1, shape=[-1, 1, self.max_len], name="s_w1"))  # b x 1 x max_len
            w1_after_attention = tf.matmul(s_w1, self.biLSTM_outputs_list[1])  # [b x 1 x max_len]*[b x max_len x 2*hidden
            self.w1_after_attention = tf.nn.dropout(w1_after_attention, self.keep_prob_placeholder)
        with tf.variable_scope("dot_layer"):
            weighted1, weighted0 =self.dot_product_attention(x_sen=self.w0_after_attention,y_sen=self.w1_after_attention,x_len = self.max_len,y_len=self.max_len)
            diff_01 = tf.subtract(self.w0_after_attention,weighted1)
            diff_10 = tf.subtract(self.w1_after_attention,weighted0)
            mul_01 = tf.multiply(self.w0_after_attention,weighted1)
            mul_10 = tf.multiply(self.w1_after_attention,weighted0)
        #cocat of w0 representation and w1 representation
        pred_info=tf.reduce_max(tf.concat([diff_01,diff_10,mul_01,mul_10],axis=2),axis=1)  #(b,4h)
        with tf.variable_scope("pred-layer"):
            fnn1 = self.fnn(input      = pred_info,
                            out_dim    = self.lstm_size,
                            activation = tf.nn.tanh,
                            use_bias   = True,
                            w_name     = "fnn-pred-W")

            if self.train_or_others:
                fnn1 = tf.nn.dropout(fnn1, self.keep_prob_placeholder)

            W_pred = tf.get_variable("W_pred", shape=[self.lstm_size, 2],regularizer=tf.contrib.layers.l2_regularizer(self.l2_strength))
            self.pred_info = tf.nn.softmax(tf.matmul(fnn1, W_pred), name="pred")


    def pred_loss(self):

        self.dense_output = tf.layers.dense(self.pred_info,2)
        sig_pred = tf.sigmoid(self.dense_output)
        onehot_true_label = tf.one_hot(self.input_label,2)
        print(" tf.shape(self.dense_output):",tf.shape(self.dense_output))
        print(" tf.shape(onehot_true_label):",tf.shape(onehot_true_label))
        crossent = tf.nn.sigmoid_cross_entropy_with_logits(labels = onehot_true_label,logits = self.dense_output)
        batch_size = tf.shape(self.input_label)[0]
        self.loss = tf.reduce_sum(crossent)/tf.cast(batch_size,tf.float32)
        self.pred = tf.argmax(sig_pred,1)
        if not self.train_or_others:
            return
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.max_grad_norm)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))

    def batch_fit(self,session, w0, w1, r, c, d, label,len_w0, len_w1, len_r, len_c, len_d, dropout_rate):
        input_feed = {self.inputs_w0:w0,
                      self.inputs_w1:w1,
                      self.inputs_r:r,
                      self.inputs_c:c,
                      self.inputs_d:d,
                      self.input_label:label,
                      self.input_length[0]:len_w0,
                      self.input_length[1]:len_w1,
                      self.input_length[2]:len_r,
                      self.input_length[3]:len_c,
                      self.input_length[4]:len_d,
                      self.keep_prob_placeholder: dropout_rate
                      }
        output_feed = [self.optimizer, self.loss, self.pred]
        outputs = session.run(output_feed, input_feed)
        return outputs[0], outputs[1], outputs[2]

    def batch_dev(self,session, w0, w1, r, c, d, label,len_w0, len_w1, len_r, len_c, len_d, dropout_rate):
        input_feed = {self.inputs_w0:w0,
                      self.inputs_w1:w1,
                      self.inputs_r:r,
                      self.inputs_c:c,
                      self.inputs_d:d,
                      self.input_label:label,
                      self.input_length[0]:len_w0,
                      self.input_length[1]:len_w1,
                      self.input_length[2]:len_r,
                      self.input_length[3]:len_c,
                      self.input_length[4]:len_d,
                      self.keep_prob_placeholder: dropout_rate
                      }
        output_feed = [self.loss, self.pred]
        outputs = session.run(output_feed, input_feed)
        return outputs[0], outputs[1]

    def batch_test(self,session, w0, w1, r, c, d, len_w0, len_w1, len_r, len_c, len_d, dropout_rate):
        input_feed = {self.inputs_w0:w0,
                      self.inputs_w1:w1,
                      self.inputs_r:r,
                      self.inputs_c:c,
                      self.inputs_d:d,
                      self.input_length[0]:len_w0,
                      self.input_length[1]:len_w1,
                      self.input_length[2]:len_r,
                      self.input_length[3]:len_c,
                      self.input_length[4]:len_d,
                      self.keep_prob_placeholder: dropout_rate
                      }
        output_feed = [self.pred]
        outputs = session.run(output_feed, input_feed)
        return outputs[0]

    def fnn(self,input,out_dim,in_dim=None,activation=None,use_bias=False,w_name="fnn-W"):
        with tf.variable_scope("fnn-layer"):
            if in_dim==None:
                input_shape = input.get_shape().as_list()
                in_dim = input_shape[-1]

            W = tf.get_variable(w_name,shape=[in_dim,out_dim])
            out = tf.matmul(input,W)

            if use_bias == True:
                b_name = w_name + '-b'
                b = tf.get_variable(b_name, shape=[out_dim])
                out = out + b

            if activation is not None:
                out = activation(out)
        return out

