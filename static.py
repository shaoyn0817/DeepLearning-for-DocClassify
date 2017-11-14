#-*- coding:utf-8 -*-
import numpy as np
import gensim
import tensorflow as tf
import math
import pandas as pd
import matplotlib.pyplot as plt
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

titledict = {}
contentdict = {}

def process(x):
    str = x['title']
    str = str.strip().split(' ')
    length = len(str)
    if titledict.has_key(length):
        titledict[length] = titledict[length]+1
    else:
        titledict[length] = 1

    if isinstance(x['content'], float):
        x['content'] = ''
    str = x['content']
    str = str.strip().split(' ')
    length = len(str)
    if contentdict.has_key(length):
        contentdict[length] = contentdict[length]+1
    else:
        contentdict[length] = 1
    return x

data = pd.read_excel('/home/shaoyn/BDCI2017-360/data/'+'train'+'data.xlsx')
data = data.apply(process, axis=1)
titlekey = list(titledict)
titlekey.sort()
contentkey = list(contentdict)
contentkey.sort()

titlevalue = [titledict[x] for x in titlekey]
contentvalue = [contentdict[x] for x in contentkey]
print contentdict
print contentkey
print contentvalue

plt.subplot(1,2,1)
plt.plot(titlekey, titlevalue)
plt.subplot(1,2,2)
plt.plot(contentkey, contentvalue)
plt.show()

