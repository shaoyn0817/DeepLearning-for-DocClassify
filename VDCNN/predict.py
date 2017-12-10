#coding:utf8
from __future__ import division
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import sys
import shutil
import time
import network
import config as cfg
import datetime
import gc
from time import strftime
import pickle

sys.path.insert(0, '../../')
from utils import to_categorical,score_eval,labellist2id,write_submission,get_id_list
from Dataset import Dataset
preFolder='../../'
settings = network.Settings()
title_len = settings.title_len
ckpt_path = settings.ckpt_path
model_name= settings.model_name

local_scores_path = preFolder+cfg.file_local_scores_path
scores_path =preFolder+cfg.file_scores_path
sub_path=preFolder+cfg.file_sub_path
if not os.path.exists(local_scores_path):
    os.makedirs(local_scores_path)
if not os.path.exists(scores_path):
    os.makedirs(scores_path)
if not os.path.exists(sub_path):
    os.makedirs(sub_path)
embedding_path = preFolder+cfg.file_embedding_npy
id_list_path=preFolder+cfg.file_test_fenci

n_tr_batches=400000/cfg.BATCH_SIZE+1
n_va_batches=100000/cfg.BATCH_SIZE+1
#测试
#n_te_batches =10

with open(preFolder+cfg.file_sr_label2id, 'rb') as inp:
        sr_title2id = pickle.load(inp)
        sr_id2title = pickle.load(inp)





def local_predict(sess, model):
    """Test on the valid data."""
    starttime = datetime.datetime.now()
    dataset=Dataset(shuffle=False)
    predict_labels_list = list()  # 所有的预测结果
    marked_labels_list = list()
    predict_scores = list()
    batch_iter=dataset.batch_iter(TrainOrValid=False)
    for i,(X1_batch,X2_batch,y_batch) in tqdm(enumerate(batch_iter)):
        marked_labels_list.extend(y_batch)
        _batch_size = len(X1_batch)
        fetches = [model.y_pred]
        feed_dict = {model.X1_inputs: X1_batch, model.X2_inputs: X2_batch,
                     model.batch_size: _batch_size, model.is_training: False, model.keep_prob: 1.0}
        predict_labels = sess.run(fetches, feed_dict)[0]
        predict_scores.append(predict_labels)
        predict_labels = map(labellist2id, predict_labels)
        predict_labels_list.extend(predict_labels)
    predict_label_and_marked_label_list = zip(predict_labels_list, marked_labels_list)
    precision, recall, f1 = score_eval(predict_label_and_marked_label_list)
    print('Local valid p=%g, r=%g, f1=%g' % (precision, recall, f1))
    predict_scores = np.vstack(np.asarray(predict_scores))
    local_scores_name = local_scores_path + model_name + '.npy'
    np.save(local_scores_name, predict_scores)
    print('local_scores.shape=', predict_scores.shape)
    print('Writed the scores into %s' 
          % (local_scores_name))
    print ('cost time:',datetime.datetime.now()-starttime )
    

    
def predict(sess, model):
    """Test on the test data."""
    starttime = datetime.datetime.now()
    dataset=Dataset(test=True)
    predict_scores = list()
    predict_labels_list = list()
    batch_iter=dataset.batch_test_iter()
    for i,(X1_batch,X2_batch) in tqdm(enumerate(batch_iter)):
        if i%1000==0:
            print ('iter:%d',i)
        _batch_size = len(X1_batch)
        fetches = [model.y_pred]
        feed_dict = {model.X1_inputs: X1_batch, model.X2_inputs: X2_batch,
                     model.batch_size: _batch_size, model.is_training: False, model.keep_prob: 1.0}
        predict_labels = sess.run(fetches, feed_dict)[0]
        predict_scores.append(predict_labels)
        predict_labels = map(labellist2id, predict_labels)
        predict_labels_list.extend(predict_labels)
    predict_scores = np.vstack(np.asarray(predict_scores))
    scores_name = scores_path + model_name + '.npy'
    np.save(scores_name, predict_scores)
    print('scores.shape=', predict_scores.shape)
    print('Writed the scores into %s' 
          % (scores_name))
    print ('cost time:%s',datetime.datetime.now()-starttime )
    return predict_labels_list

def main():
    if not os.path.exists(ckpt_path + 'checkpoint'):
        print('there is not saved model, please check the ckpt path')
        exit()
    print('Loading model...')
    W_embedding = np.load(embedding_path)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = network.TextCNN(W_embedding, settings)
        model.saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
        #print('valid2 predicting...')
        #print ('valid batches:%d'%n_va_batches)
        #valid2_predict(sess, model)
        print('Local predicting...')
        print ('valid batches:%d'%n_va_batches)
        local_predict(sess, model)
        print('Test predicting...')
        print ('test batches:%d'%n_tr_batches)
        results=predict(sess, model)
        sub_path_name=sub_path+model_name+str(strftime("%m%d%H%M"))+'.csv'
        id_list=get_id_list(id_list_path)
        write_submission(sub_path_name,id_list,results,sr_id2title)

if __name__ == '__main__':
    main()
