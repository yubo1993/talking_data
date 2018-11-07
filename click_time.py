# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 15:46:09 2018

@author: 11497
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import os
#print(os.listdir("../input"))
import gc

import time

import matplotlib.pyplot as plt
import seaborn as sns


# 读取数据
train_data = pd.read_csv(r'C:\Users\11497\Desktop\train_sample.csv', parse_dates=['click_time'])
test_data = pd.read_csv(r'C:\Users\11497\Desktop\test.csv', parse_dates=['click_time'])

# 将时间分解
train_data['day'] = train_data['click_time'].dt.day.astype('uint8')
train_data['hour'] = train_data['click_time'].dt.hour.astype('uint8')
train_data['minute'] = train_data['click_time'].dt.minute.astype('uint8')
train_data['second'] = train_data['click_time'].dt.second.astype('uint8')


train_1 = train_data.groupby(['ip', 'app', 'device', 'os']).agg({'click_time': 'count'}).reset_index().rename(index=str, columns={'click_time': 'ct1_count'})
train_2 = train_data.groupby(['ip', 'device', 'os', 'channel']).agg({'click_time': 'count'}).reset_index().rename(index=str, columns={'click_time': 'ct2_count'})
train_3 = train_data.groupby(['ip', 'app', 'device', 'os', 'channel']).agg({'click_time': 'count'}).reset_index().rename(index=str, columns={'click_time': 'ct3_count'})
train = train_data.copy()
# 合并

train = train.merge(train_1, on=['ip', 'app', 'device', 'os'], how='left')
train = train.merge(train_2, on=['ip', 'device', 'os', 'channel'], how='left')
train = train.merge(train_3, on=['ip', 'app', 'device', 'os', 'channel'], how='left')





