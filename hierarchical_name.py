import tensorflow as tf
import numpy as np
import pandas as pd

class hierarchical_model():
    def __int__(self, hidden_size, num_label):
        self.hidden_size = hidden_size
        self.num_label = num_label
        print 'Initialize a hierachical attention model...'
        self.weights = {
            'atten_w1':tf.get_variable(name='atten_w1', shape=[self.hidden_size, 1.5*self.hidden_size], initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32)),
            'atten_w2':tf.get_variable(name='atten_w2', shape=[1.5*self.hidden_size, 1.5*self.hidden_size], initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32)),
            'fc_w3':tf.get_variable(name='fc_w3', shape=[1.5*self.hidden_size, 1], initializer=tf.contrib.layers.xavier_initializer(tf.float32)),
            'glob_w1':tf.get_variable(name='glob_w1', shape=[hidden_size, num_label], initializer=tf.contrib.layers.xavier_initializer(tf.float32)),
            #tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
        }

        self.bias = {
            'atten_b1':tf.get_variable(name='atten_b1', shape=[1.5*self.hidden_size,], initializer=tf.constant_initializer(value=0, dtype=tf.float32)),
            'atten_b2':tf.get_variable(name='atten_b2', shape=[1.5*self.hidden_size,], initializer=tf.constant_initializer(value=0, dtype=tf.float32)),
            'fc_b3':tf.get_variable(name='fc_b3', shape=[1.5*self.hidden_size], initializer=tf.constant_initializer(value=0, dtype=tf.float32)),
            'glob_b1':tf.get_variable(name='glob_b1', shape=[hidden_size,], initializer=tf.constant_initializer(value=0, dtype=tf.float32)),
        }

    def word_rnn(self, X, keep_prob, num_layer,batch_size, istrain):

        cell = tf.contrib.BasicLSTMCell(num_units=self.hidden_size)

        if istrain:
            cell = tf.contrib.DropoutWrapper(cell=cell, output_keep_prob=keep_prob)
        cell = tf.contrib.MultiRNNCell([cell]*num_layer)
        init_state = cell.zero_state(batch_size, dtype=tf.float32)
        output, state = tf.nn.dynamic_rnn(cell, inputs=X, initial_state=init_state, time_major=False)
        #output size  [batchsize, timestep_size, hiddensize]
        #h_state = output[:, -1, :]
        atten1 = tf.matmul(output, self.weights['atten_w1'])+self.bias['atten_b1']
        atten2 = tf.matmul(atten1, self.weights['atten_w2'])+self.bias['atten_b2']
        atten = tf.matmul(atten2, self.weights['fc_w3'])+self.bias['fc_b3']
        #atten shape [batchsize, timestep_size, 1]
        atten = tf.squeeze(atten, [2])  #atten shape [batchsize, timestep_size]
        a = tf.nn.softmax(atten, 1)
        #[4, 3, 1]   [4, 3, 5]
        sen_emb = a*output
        sen_emb = tf.reduce_sum(sen_emb, 1)
        return  sen_emb

    def sen_rnn(self, X, keep_prob, num_layer,batch_size, istrain):

        cell = tf.contrib.BasicLSTMCell(num_units=self.hidden_size)
        if istrain:
            cell = tf.contrib.DropoutWrapper(cell=cell, output_keep_prob=keep_prob)
        cell = tf.contrib.MultiRNNCell([cell]*num_layer)
        init_state = cell.zero_state(batch_size, dtype=tf.float32)
        output, state = tf.nn.dynamic_rnn(cell, inputs=X, initial_state=init_state, time_major=False)
        #output size  [batchsize, timestep_size, hiddensize]
        #h_state = output[:, -1, :]
        atten1 = tf.matmul(output, self.weights['atten_w1'])+self.bias['atten_b1']
        atten2 = tf.matmul(atten1, self.weights['atten_w2'])+self.bias['atten_b2']
        atten = tf.matmul(atten2, self.weights['fc_w3'])+self.bias['fc_b3']
        #atten shape [batchsize, timestep_size, 1]
        atten = tf.squeeze(atten, [2])  #atten shape [batchsize, timestep_size]
        a = tf.nn.softmax(atten, 1)
        #[4, 3, 1]   [4, 3, 5]
        doc_emb = a*output
        doc_emb = tf.reduce_sum(doc_emb, 1)
        return  doc_emb  #[batchsize, hiddensize]


    def inference(self, X, keep_prob, num_layer,batch_size):
        sen_emb = self.word_cnn(X, keep_prob, num_layer,batch_size, True)
        doc_emb = self.sen_rnn(sen_emb, keep_prob, num_layer,batch_size, True)
        # [batchsize, hiddensize]
        out = tf.matmul(doc_emb, self.weights['glob_w1'])+self.bias['glob_b1']
        dense = tf.nn.relu(out)
        return dense

    def soft_max(self, dense, labels):
        predicts = tf.nn.softmax(dense)
        labels = tf.one_hot(labels, self.num_label)
        loss = -tf.reduce_mean(labels*tf.log(predicts+1e-6))
        return loss

    def inference_test(self, X, keep_prob, num_layer,batch_size):
        sen_emb = self.word_cnn(X, keep_prob, num_layer,batch_size, False)
        doc_emb = self.sen_rnn(sen_emb, keep_prob, num_layer,batch_size, False)
        # [batchsize, hiddensize]
        out = tf.matmul(doc_emb, self.weights['glob_w1'])+self.bias['glob_b1']
        dense = tf.nn.relu(out)
        return dense

    def get_acc(self, dense, label):
        p = tf.nn.softmax(dense)
        prediction = tf.cast(tf.arg_max(p, 1), tf.int32)
        correct_prediction = tf.equal(prediction, label)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return acc

    def optimizer(self, loss, lr):
        train_opt = tf.train.AdamOptimizer(lr).minimize(loss)
        return train_opt