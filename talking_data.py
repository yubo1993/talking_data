# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 16:30:53 2018

@author: YUBO
"""
import pandas as pd
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
#加载数据集
train_data=pd.read_csv("train.csv",dtype=dtypes,usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
test_data=pd.read_csv("test.csv",dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time','click_id'])
sub = pd.DataFrame()
sub['click_id'] = test_data['click_id']
train_df=train_data.append(test_data)
len_train = len(train_data)
del test_data
#看看各类别与ip大小是否存在关系
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16,4))
bins = 20
ax1.hist(train_data["ip"][train_data["is_attributed"]== 1], bins = bins)
ax1.set_title('Fraud')
ax2.hist(train_data["ip"][train_data["is_attributed"]== 0], bins = bins)
ax2.set_title('Normal')
plt.xlabel('ip')
plt.ylabel('count')
plt.show()
#这里要看看以hour做groupby的点击次数分布，找关系
train_data['hour'] = pd.to_datetime(train_data.click_time).dt.hour.astype('uint8')
sns.factorplot(x="hour", data=train_data.ix[train_data["is_attributed"]==0,:], kind="count",  palette="ocean", size=6, aspect=3)
sns.factorplot(x="hour", data=train_data.ix[train_data["is_attributed"]==1,:], kind="count",  palette="ocean", size=6, aspect=3)

gc.collect()
#依次做特征组合
#分组点击次数的衍生特征（按照click_time衍生）
# 将时间分解
train_df['day'] =  pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
train_df['hour'] =  pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['minute'] =  pd.to_datetime(train_df.click_time).dt.minute.astype('uint8')
train_df['second'] =  pd.to_datetime(train_df.click_time).dt.second.astype('uint8')
gp = train_df.groupby(['ip', 'app', 'device', 'os']).agg({'click_time': 'count'}).reset_index().rename(index=str, columns={'click_time': 'ct1_count'})
train_df = train_df.merge(gp, on=['ip', 'app', 'device', 'os'], how='left')
del gp
gc.collect()
gp = train_df.groupby(['ip', 'device', 'os', 'channel']).agg({'click_time': 'count'}).reset_index().rename(index=str, columns={'click_time': 'ct2_count'})
train_df= train_df.merge(gp, on=['ip', 'device', 'os', 'channel'], how='left')
del gp
gc.collect()
gp = train_df.groupby(['ip', 'app', 'device', 'os', 'channel']).agg({'click_time': 'count'}).reset_index().rename(index=str, columns={'click_time': 'ct3_count'})
train_df= train_df.merge(gp, on=['ip', 'app', 'device', 'os', 'channel'], how='left')
del gp
gc.collect()
#基于ip和device组合的其他特征的count和unique
gp = train_df.groupby(['ip', 'device']).agg({'app': 'unique'}).reset_index().rename(index=str, columns={'app': 'ct4_count'})
train_df= train_df.merge(gp, on=['ip', 'device'], how='left')
del gp
gc.collect()
count_app=[]
for s in train_df['ct4_count']:
    count_app.append(len(s))
train_df['ct4_count']=count_app##每个用户浏览app数目
del count_app
gp = train_df.groupby(['ip', 'device']).agg({'os': 'unique'}).reset_index().rename(index=str, columns={'os': 'ct5_count'})
train_df= train_df.merge(gp, on=['ip', 'device'], how='left')
del gp
gc.collect()
count_os=[]
for s in train_df['ct5_count']:
    count_os.append(len(s))
train_df['ct5_count']=count_os##每个用户对应os次数
del count_os
gp = train_df.groupby(['ip', 'device']).agg({'channel': 'unique'}).reset_index().rename(index=str, columns={'channel': 'ct6_count'})
train_df= train_df.merge(gp, on=['ip', 'device'], how='left')
del gp
gc.collect()
count_channel=[]
for s in train_df['ct6_count']:
    count_channel.append(len(s))
train_df['ct6_count']=count_channel##每个用户对应channel次数
del count_channel
#仅对ip分组，对其他特征做count和unique
gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_t_cha_count'})
train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
del gp
gc.collect()
gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].nunique().reset_index().rename(index=str, columns={'channel': 'ip_t_cha_unique'})
train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
del gp
gc.collect()
gp = train_df[['ip','day','hour','app']].groupby(by=['ip','day','hour'])[['app']].count().reset_index().rename(index=str, columns={'app': 'ip_t_app_count'})
train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
del gp
gc.collect()
gp = train_df[['ip','day','hour','app']].groupby(by=['ip','day','hour'])[['app']].nunique().reset_index().rename(index=str, columns={'app': 'ip_t_app_unique'})
train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
del gp
gc.collect()
gp = train_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_cha_count'})
train_df = train_df.merge(gp, on=['ip','app'], how='left')
del gp
gc.collect()
gp = train_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].nunique().reset_index().rename(index=str, columns={'channel': 'ip_app_cha_unique'})
train_df = train_df.merge(gp, on=['ip','app'], how='left')
del gp
gc.collect()
gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
del gp
gc.collect()
gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','channel'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_tchan_hvar'})
train_df = train_df.merge(gp, on=['ip','day','channel'], how='left')
del gp
gc.collect()
gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','channel'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_tchan_hmean'})
train_df = train_df.merge(gp, on=['ip','day','channel'], how='left')
del gp
gc.collect()
gp = train_df[['ip','day','hour','app']].groupby(by=['ip','day','app'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_tapp_hvar'})
train_df = train_df.merge(gp, on=['ip','day','app'], how='left')
del gp
gc.collect()
gp = train_df[['ip','day','hour','app']].groupby(by=['ip','day','app'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_tapp_hmean'})
train_df = train_df.merge(gp, on=['ip','day','app'], how='left')
del gp
gc.collect()
gp = train_df[['ip','app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])[['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
del gp
gc.collect()
gp = train_df[['ip','app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])[['day']].mean().reset_index().rename(index=str, columns={'day': 'ip_app_channel_mean_day'})
train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
del gp
gc.collect()
gp = train_df[['ip','app', 'channel','hour']].groupby(by=['ip', 'app', 'channel'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'})
train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
del gp
gc.collect()
gp = train_df[['ip','app', 'channel','hour']].groupby(by=['ip', 'app', 'channel'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_var_hour'})
train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
del gp
gc.collect()
gp = train_df[['ip','app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_app_os_var'})
train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
del gp
gc.collect()
gp = train_df[['ip','app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_os_mean'})
train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
del gp
gc.collect()
#train_df.to_csv("train_df.csv",index=False)
#数据读取
train_df=pd.read_csv("train_df.csv")
#train_df.info()
train_df['ct1_count'] = train_df['ct1_count'].astype('uint16')
train_df['ct2_count'] = train_df['ct2_count'].astype('uint16')
train_df['ct3_count'] = train_df['ct3_count'].astype('uint16')
train_df['ct4_count'] = train_df['ct4_count'].astype('uint16')
train_df['ct5_count'] = train_df['ct5_count'].astype('uint16')
train_df['ct6_count'] = train_df['ct6_count'].astype('uint16')
train_df['ip_t_cha_count'] = train_df['ip_t_cha_count'].astype('uint16')
train_df['ip_t_cha_unique'] = train_df['ip_t_cha_unique'].astype('uint16')
train_df['ip_t_app_count'] = train_df['ip_t_app_count'].astype('uint16')
train_df['ip_t_app_unique'] = train_df['ip_t_app_unique'].astype('uint16')
train_df['ip_app_cha_count'] = train_df['ip_app_cha_count'].astype('uint16')
train_df['ip_app_cha_unique'] = train_df['ip_app_cha_unique'].astype('uint16')
train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')
train_df['ip_tchan_hmean']=train_df['ip_tchan_hmean'].astype('uint16')
train_df['ip_tapp_hmean']=train_df['ip_tapp_hmean'].astype('uint16')
train_df['ip_app_channel_mean_day']=train_df['ip_app_channel_mean_day'].astype('uint16')
train_df['ip_app_channel_mean_hour']=train_df['ip_app_channel_mean_hour'].astype('uint16')
train_df['ip_app_os_mean']=train_df['ip_app_os_mean'].astype('uint16')
#分割处理完之后的训练集测试集
test_data = train_df[len_train:]
train_data = train_df[:(len_train-6000000)]
val_data = train_df[(len_train-6000000):len_train]
del train_df
#建立lgb模型
#train_df.info()
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.01,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 8,
        'verbose': 0,
        
        'tree_learner':'data'
    }

    lgb_params.update(params)

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

    bst1 = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=feval)

    n_estimators = bst1.best_iteration
    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics+":", evals_results['valid'][metrics][n_estimators-1])

    return bst1

target = 'is_attributed'
predictors = ['app', 'channel', 'device', 'ip', 'os',
       'day', 'hour', 'minute', 'second', 'ct1_count', 'ct2_count',
       'ct3_count', 'ct4_count', 'ct5_count', 'ct6_count', 'ip_t_cha_count',
       'ip_t_cha_unique', 'ip_t_app_count', 'ip_t_app_unique',
       'ip_app_cha_count', 'ip_app_cha_unique', 'ip_app_os_count',
       'ip_tchan_hvar', 'ip_tchan_hmean', 'ip_tapp_hvar', 'ip_tapp_hmean',
       'ip_app_channel_var_day', 'ip_app_channel_mean_day',
       'ip_app_channel_mean_hour', 'ip_app_channel_var_hour', 'ip_app_os_var',
       'ip_app_os_mean']
categorical = ['app','device','os', 'channel','ip']

params = {
    'learning_rate': 0.1,
    'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
    'max_depth': 3,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight':400 # because training data is extremely unbalanced 
}
bst = lgb_modelfit_nocv(params, 
                        train_data, 
                        val_data, 
                        predictors, 
                        target, 
                        objective='binary', 
                        metrics='auc',
                        early_stopping_rounds=50, 
                        verbose_eval=True, 
                        num_boost_round=300, 
                        categorical_features=categorical)

del train_data
del val_data
gc.collect()

print("Predicting...")
sub['is_attributed'] = bst.predict(test_data[predictors])
print("writing...")
sub.to_csv('sub_lgb_balanced99.csv',index=False)
print("done...")
print(sub.info())
