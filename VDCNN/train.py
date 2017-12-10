# -*- coding:utf-8 -*-
from __future__ import division
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import sys
import shutil
import time
import network as network
import config as cfg
import datetime
import gc

sys.path.insert(0, '../../')
from Dataset import Dataset
from utils import to_categorical,score_eval,labellist2id

preFolder='../../'
flags = tf.flags
flags.DEFINE_bool('is_retrain', False, 'if is_retrain is true, not rebuild the summary')
flags.DEFINE_integer('max_epoch', 2, 'update the embedding after max_epoch, default: 1')
flags.DEFINE_integer('max_max_epoch', 150, 'all training epoches, default: 6')
flags.DEFINE_float('lr', 0.0001, 'initial learning rate, default: 8e-4') 
flags.DEFINE_float('keep_prob', 1, 'keep_prob for training, default: 0.5')
# 正式
flags.DEFINE_integer('valid_step', 400, 'valid_step, default: 10000')
flags.DEFINE_float('last_f1', 0.75, 'if valid_f1 > last_f1, save new model. default: 0.80')

FLAGS = flags.FLAGS


last_f1 = FLAGS.last_f1
settings = network.Settings()
title_len = settings.title_len
content_len = settings.content_len
#summary_path = settings.summary_path
ckpt_path = settings.ckpt_path
model_path = ckpt_path + 'model.ckpt'
embedding_path = preFolder+cfg.file_embedding_npy

n_tr_batches=500000/cfg.BATCH_SIZE+1
n_va_batches=100000/cfg.BATCH_SIZE+1


def valid_epoch(dataset, sess, model):
    """Test on the valid data."""
    batch_iter=dataset.batch_iter(False)
    _costs = 0.0
    predict_labels_list = list()  # 所有的预测结果
    num = 0
    marked_labels_list = list()
    for  i,(X1_batch,X2_batch,y_batch) in enumerate(batch_iter):
        num = num + 1
        marked_labels_list.extend(y_batch)
        y_batch = to_categorical(y_batch)
        _batch_size = len(y_batch)
        fetches = [model.loss, model.y_pred]
        feed_dict = {model.X1_inputs: X1_batch, model.X2_inputs: X2_batch, model.y_inputs: y_batch,
                     model.batch_size: _batch_size, model.is_training: False, model.keep_prob: 1.0}
        _cost, predict_labels = sess.run(fetches, feed_dict)
        _costs += _cost
        predict_labels = map(labellist2id, predict_labels)
        predict_labels_list.extend(predict_labels)
    predict_label_and_marked_label_list = zip(predict_labels_list, marked_labels_list)
    precision, recall, f1 = score_eval(predict_label_and_marked_label_list)
    mean_cost = _costs / num
    return mean_cost, precision, recall, f1

def valid_train_epoch(dataset, sess, model, train_sample):
    """Test on the train data."""
    batch_iter=train_sample.train_sample_iter()
    _costs = 0.0
    num = 0
    marked_labels_list = list()
    for  i,(X1_batch,X2_batch,y_batch) in enumerate(batch_iter):
        num = num + 1
        marked_labels_list.extend(y_batch)
        y_batch = to_categorical(y_batch)
        _batch_size = len(y_batch)
        fetches = [model.loss]
        feed_dict = {model.X1_inputs: X1_batch, model.X2_inputs: X2_batch, model.y_inputs: y_batch,
                     model.batch_size: _batch_size, model.is_training: False, model.keep_prob: 1.0}
        _cost = sess.run(fetches, feed_dict)
        _costs += _cost[0]

    mean_cost = _costs / num
    return mean_cost

def train_epoch(dataset, sess, model, train_fetches, valid_fetches, train_writer, test_writer, train_sample):
    global last_f1
    starttime = datetime.datetime.now()
    batch_iter=dataset.batch_iter()
    for i,(X1_batch,X2_batch,y_batch) in enumerate(batch_iter):
        # training
        y_batch = to_categorical(y_batch)
        _batch_size = len(y_batch)
        feed_dict = {model.X1_inputs: X1_batch, model.X2_inputs: X2_batch, model.y_inputs: y_batch,
                     model.batch_size: _batch_size, model.is_training: True, model.keep_prob: 0.8}
        sess.run(train_fetches, feed_dict)  # the cost is the mean cost of one batch
        global_step = sess.run(model.global_step)
        if 0 == (global_step) % FLAGS.valid_step:
            valid_cost, precision, recall, f1 = valid_epoch(dataset, sess, model)
            train_sample_cost = valid_train_epoch(dataset, sess, model, train_sample)
            print()
            print('Global_step=%d: p=%g, r=%g, || f1=%g, valid loss=%g, train loss=%g' % (
                global_step, precision, recall, f1, valid_cost, train_sample_cost))
            if f1 > last_f1 :
                last_f1 = f1
                saving_path = model.saver.save(sess, model_path+str(f1)+'_', (global_step))
                print('saved new model to %s ' % saving_path)
            print()
             

def main(_):
    global ckpt_path
    global last_f1
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    firsttime = datetime.datetime.now()
    print('1.Loading data...')
    W_embedding = np.load(embedding_path)
    print('training sample_num = %d' % n_tr_batches)
    print('valid sample_num = %d' % n_va_batches)

    # Initial or restore the model
    print('2.Building model...')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = network.TextCNN(W_embedding, settings)
        with tf.variable_scope('training_ops') as vs:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                global_step = model.global_step
                optimizer = tf.train.MomentumOptimizer(FLAGS.lr, 0.9)
                train_op = tf.contrib.layers.optimize_loss(loss=model.loss, global_step=global_step, clip_gradients=4.0,
                        learning_rate=FLAGS.lr, optimizer=optimizer, update_ops=update_ops)

            training_ops = [v for v in tf.global_variables() if v.name.startswith(vs.name+'/')]

        dataset=Dataset(cross_validation=3)
        train_sample=Dataset(valid=True, cross_validation=3)
        # 如果已经保存过模型，导入上次的模型
        if os.path.exists(ckpt_path + "checkpoint"):
            print("Restoring Variables from Checkpoint...")
            model.saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
            last_valid_cost, precision, recall, last_f1 = valid_epoch(dataset, sess, model)
            print(' valid cost=%g; p=%g, r=%g, f1=%g' % (last_valid_cost, precision, recall, last_f1))
            sess.run(tf.variables_initializer(training_ops))
        else: 
            print('Initializing Variables...')
            sess.run(tf.global_variables_initializer())

        print('3.Begin training...')
        for epoch in range(FLAGS.max_max_epoch):
            print ('epoch %d ********************************************************'%epoch)
            starttime = datetime.datetime.now()
            train_fetches = [train_op]
            valid_fetches = [model.loss]
            train_epoch(dataset, sess, model, train_fetches, valid_fetches, '', '', train_sample)
            print ('epoch %d cost time:%s'%(epoch, datetime.datetime.now()-starttime))
            print ('total cost time:%s'%(datetime.datetime.now()-firsttime))
        # 最后再做一次验证
        valid_cost, precision, recall, f1 = valid_epoch(dataset, sess, model)
        print('END.Global_step=%d: valid cost=%g; p=%g, r=%g, f1=%g' % (
            sess.run(model.global_step), valid_cost, precision, recall, f1))
        if f1 > last_f1:  # save the better model
            saving_path = model.saver.save(sess, model_path+str(f1)+'_', sess.run(model.global_step))
            print('saved new model to %s ' % saving_path)


if __name__ == '__main__':
    tf.app.run()
