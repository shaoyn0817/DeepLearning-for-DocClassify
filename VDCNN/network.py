# -*- coding:utf-8 -*-
import sys
sys.path.insert(0, '../../')
import tensorflow as tf
import config as cfg
import numpy as np

"""wd_1_1_cnn_concat
title 部分使用 TextCNN；content 部分使用 TextCNN； 两部分输出直接 concat。
"""
preFolder='../../'

class Settings(object):
    def __init__(self):
        self.model_name = 'textCNN_VD_final'
        self.title_len = cfg.PAD_TITLE_LEEHTH
        self.content_len = cfg.PAD_CONTENT_LEENTH
        self.block_num=4
        self.use_k_max_pooling=False
        self.n_class = 2
        self.l2_reg_lambda=0
        self.summary_path = preFolder+cfg.file_summary_path + self.model_name + '/'
        self.ckpt_path =preFolder+cfg.file_ckpt_path + self.model_name + '/'


class TextCNN(object):
    """
    title: inputs->textcnn->output_title
    content: inputs->textcnn->output_content
    concat[output_title, output_content] -> fc+bn+relu -> sigmoid_entropy.
    """

    def __init__(self, W_embedding, settings):
        self.model_name = settings.model_name
        self.title_len = settings.title_len
        self.content_len = settings.content_len
        self.block_num=settings.block_num
        self.l2_reg_lambda=settings.l2_reg_lambda
        self.use_k_max_pooling=settings.use_k_max_pooling
        self.n_class = settings.n_class
        
        self._global_step = tf.Variable(0, trainable=False, name='Global_Step')
        self.update_emas = list()
        self.conv_initializer = tf.contrib.layers.xavier_initializer()
        self.linear_initializer = tf.contrib.layers.xavier_initializer()
        # placeholders
        self._is_training =  tf.placeholder(tf.bool)
        self._keep_prob = tf.placeholder(tf.float32, [])
        self._batch_size = tf.placeholder(tf.int32, [])

        with tf.name_scope('Inputs'):
            self._X1_inputs = tf.placeholder(tf.int64, [None, self.title_len], name='X1_inputs')
            self._X2_inputs = tf.placeholder(tf.int64, [None, self.content_len], name='X2_inputs')
            self._y_inputs = tf.placeholder(tf.float32, [None, self.n_class], name='y_input')
            
        with tf.name_scope('l2_loss'):
            self.l2_loss=tf.constant(0.0)
            
        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable(name='embedding', shape=W_embedding.shape,
                                             initializer=tf.constant_initializer(W_embedding), trainable=True)
        self.embedding_size = W_embedding.shape[1]

        #vdcnn_title
        #with tf.variable_scope('vdcnn_title'):
         #   output_title = self.VDCnn_inference(self._X1_inputs)

        with tf.variable_scope('vdcnn_content'):
            output_content = self.VDCnn_inference(self._X2_inputs)

        with tf.variable_scope('fc-layer'):
            output = output_content
            #output = tf.concat([output_title, output_content], axis=1)
            self._y_pred=self.fc_inference(output)

        with tf.name_scope('loss'):
            self.predictions = tf.argmax(self._y_pred, 1, name="predictions")
            self._loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self._y_pred, labels=self._y_inputs))#+ self.l2_reg_lambda * self.l2_loss
            tf.summary.scalar('loss', self._loss)
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self._y_inputs, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        self.saver = tf.train.Saver(max_to_keep=2)

    @property
    def is_training(self):
        return self._is_training

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def global_step(self):
        return self._global_step

    @property
    def X1_inputs(self):
        return self._X1_inputs

    @property
    def X2_inputs(self):
        return self._X2_inputs

    @property
    def y_inputs(self):
        return self._y_inputs

    @property
    def y_pred(self):
        return self._y_pred

    @property
    def loss(self):
        return self._loss

    def weight_variable(self, shape, name):
        """Create a weight variable with appropriate initialization."""
        initial = tf.constant(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def fc_inference(self,inputs):
        # fc1
        with tf.variable_scope('fc1'):
            w = tf.get_variable('w', [inputs.get_shape()[1], 512], initializer= self.linear_initializer)
            tf.summary.histogram('W_1', w)
            b = tf.get_variable('b', [512], initializer=tf.constant_initializer(0.0))
            tf.summary.histogram('b_1', b)
            #self.l2_loss += tf.nn.l2_loss(w)+tf.nn.l2_loss(b)
            out = tf.matmul(inputs, w) + b
            fc1 = tf.nn.relu(out)
            drop1 = tf.nn.dropout(fc1, self._keep_prob, name='drop1') 

        # fc2
        with tf.variable_scope('fc2'):
            w = tf.get_variable('w', [drop1.get_shape()[1], 512], initializer=self.linear_initializer)
            tf.summary.histogram('W_2', w)
            b = tf.get_variable('b', [512], initializer=tf.constant_initializer(0.0))
            tf.summary.histogram('b_2', b)
            #self.l2_loss += tf.nn.l2_loss(w)+tf.nn.l2_loss(b)
            out = tf.matmul(drop1, w) + b
            fc2 = tf.nn.relu(out)
            drop2 = tf.nn.dropout(fc2, self._keep_prob, name='drop2') 

        # fc3
        with tf.variable_scope('fc3'):
            w = tf.get_variable('w', [drop2.get_shape()[1], self.n_class], initializer=self.linear_initializer)
            tf.summary.histogram('W_3', w)
            b = tf.get_variable('b', [self.n_class], initializer=tf.constant_initializer(0.0))
            tf.summary.histogram('b_3', b)
            #self.l2_loss += tf.nn.l2_loss(w)+tf.nn.l2_loss(b)
            fc3 = tf.matmul(drop2, w) + b
        return fc3
    
    def VDCnn_inference(self, X_inputs):
        """VDCnn_inference 模型。
        Args:
            X_inputs: tensor.shape=(batch_size, n_step)
        Returns:
            title_outputs: tensor.shape=(batch_size, self.n_filter_total)
        """
        inputs = tf.nn.embedding_lookup(self.embedding, X_inputs)
        inputs = tf.expand_dims(inputs, -1)
        # First Conv Layer
        with tf.variable_scope("first_conv") as scope: 
            filter_shape = [3,  self.embedding_size, 1, 64]
            w = tf.get_variable(name='W_1', shape=filter_shape, 
                initializer=self.conv_initializer)
            conv = tf.nn.conv2d(inputs, w, strides=[1, 1, self.embedding_size, 1], padding="SAME")
            b = tf.get_variable(name='b_1', shape=[64], 
                    initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(conv, b)
            first_conv = tf.nn.relu(out)
            
        # all convolutional blocks
        conv_block_1 = self.Convolutional_Block(first_conv, num_layers=10, num_filters=64, name='1', is_training=self._is_training)
        pool1 = tf.nn.max_pool(conv_block_1, ksize=[1, 3, 1, 1], strides=[1, 2, 1, 1], padding='SAME', name="pool_1")
       
 
        conv_block_2 = self.Convolutional_Block(pool1, num_layers=10, num_filters=128, name='2', is_training=self._is_training)
        pool2 = tf.nn.max_pool(conv_block_2, ksize=[1, 3, 1, 1], strides=[1, 2, 1, 1], padding='SAME', name="pool_2")


        conv_block_3 = self.Convolutional_Block(pool2, num_layers=4, num_filters=256, name='3', is_training=self._is_training)
        pool3 = tf.nn.max_pool(conv_block_3, ksize=[1, 3, 1, 1], strides=[1, 2, 1, 1], padding='SAME', name="pool_3")


        conv_block_4 = self.Convolutional_Block(pool3, num_layers=4, num_filters=512, name='4', is_training=self._is_training)


        if self.use_k_max_pooling:
            transposed = tf.transpose(conv_block_4, [0,3,2,1])
            k_pooled = tf.nn.top_k(transposed, k=8, name='k_pool')
            reshaped = tf.reshape(k_pooled[0], (-1, 512*8))
        else:
            pool4 = tf.nn.max_pool(conv_block_4, ksize=[1, 3, 1, 1], strides=[1, 2, 1, 1], padding='SAME', name="pool_4")
            shape = int(np.prod(pool4.get_shape()[1:]))
            reshaped = tf.reshape(pool4, (-1, shape))
        return reshaped
    
    def Convolutional_Block(self,inputs, num_layers, num_filters, name, is_training):
        '''
        Convolutional Block which contains 2 Conv layers
        '''
        with tf.variable_scope("conv_block_%s" % name):
            filter_shape = [3, 1, inputs.get_shape()[3], num_filters]
            w = tf.get_variable(name='W_1', shape=filter_shape, 
                initializer=self.conv_initializer)
            b = tf.get_variable(name='b_1', shape=[num_filters], 
                    initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1], padding="SAME")
            conv = tf.nn.bias_add(conv, b)
            batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=is_training)
            out = tf.nn.relu(batch_norm)

            for i in range(2, num_layers+1):
                filter_shape = [3, 1, out.get_shape()[3], num_filters]
                w = tf.get_variable(name='W_'+str(i), shape=filter_shape, 
                    initializer=self.conv_initializer)
                b = tf.get_variable(name='b_'+str(i), shape=[num_filters], 
                        initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv2d(out, w, strides=[1, 1, 1, 1], padding="SAME")
                conv = tf.nn.bias_add(conv, b)
                batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=is_training)
                out = tf.nn.relu(batch_norm)
        return out

