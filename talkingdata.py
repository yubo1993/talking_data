# -*- coding: utf-8 -*-
"""
Created on Wed May 30 17:28:46 2018

@author: YUBO
"""

import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as fn

import pyspark.ml.feature as ft
import pyspark.mllib.stat as st
from pyspark.sql.types import StructType,IntegerType,StructField,TimestampType,DataType,LongType
spark = pyspark.sql.SparkSession.builder.appName("MyApp") \
            .config("spark.jars.packages", "Azure:mmlspark:0.12") \
            .getOrCreate()

#读取数据集
schematype1 = StructType([
    StructField("ip",IntegerType(),True),
    StructField("app",IntegerType(), True),
    StructField("device",IntegerType(),True),
    StructField("os",IntegerType(), True),
    StructField("channel",IntegerType(), True),
    StructField("click_time",TimestampType(), True),
    StructField("attributed_time",TimestampType(),True),
    StructField("is_attributed",IntegerType(), True)])
schematype2 = StructType([
    StructField("click_id",IntegerType(), True),
    StructField("ip",IntegerType(), True),
    StructField("app",IntegerType(), True),
    StructField("device",IntegerType(),True),
    StructField("os",IntegerType(), True),
    StructField("channel",IntegerType(), True),
    StructField("click_time",TimestampType(), True)])
train_data=spark.read.csv("E:/taikingdata/train.csv",header=True,schema=schematype1)
test_data=spark.read.csv("E:/taikingdata/test.csv",header=True,schema=schematype2)
train_df=train_data
#点击时间的拆分    
train_df=train_df.withColumn('hour',fn.hour('click_time').cast(IntegerType()))
train_df=train_df.withColumn('day',fn.dayofmonth('click_time').cast(IntegerType()))
train_df=train_df.withColumn('minute',fn.minute('click_time').cast(IntegerType()))
train_df=train_df.withColumn('second',fn.second('click_time').cast(IntegerType()))
    #分组点击次数的衍生特征（按照click_time衍生）
gp=train_df.groupby(['ip', 'app', 'device', 'os']).agg(fn.count('click_time').cast(IntegerType()).alias('ct1_count'))
train_df = train_df.join(gp, on=['ip', 'app', 'device', 'os'], how='left')
gp=train_df.groupby(['ip', 'device', 'os', 'channel']).agg(fn.count('click_time').cast(IntegerType()).alias('ct2_count'))
train_df = train_df.join(gp, on=['ip', 'device', 'os', 'channel'], how='left')
gp=train_df.groupby(['ip', 'app', 'device', 'os', 'channel']).agg(fn.count('click_time').cast(IntegerType()).alias('ct3_count'))
train_df = train_df.join(gp, on=['ip', 'app', 'device', 'os', 'channel'], how='left')
    #仅对ip分组，对其他特征做count和unique

gp = train_df.groupby(['ip', 'app']).agg(fn.count('channel').cast(IntegerType()).alias('ip_app_cha_count'))
train_df = train_df.join(gp, on=['ip', 'app'], how='left')
gp = train_df.groupby(['ip','app']).agg(fn.countDistinct('channel').cast(IntegerType()).alias('ip_app_cha_unique'))
train_df = train_df.join(gp, on=['ip','app'], how='left')
gp = train_df.groupby(['ip', 'app', 'os']).agg(fn.count('channel').cast(IntegerType()).alias('ip_app_os_count'))
train_df = train_df.join(gp, on=['ip', 'app', 'os'], how='left')
gp = train_df.groupby(['ip', 'app', 'os']).agg(fn.countDistinct('channel').cast(IntegerType()).alias('ip_app_os_unique'))
train_df = train_df.join(gp, on=['ip', 'app', 'os'], how='left')
gp = train_df.groupby(['ip', 'app', 'channel']).agg(fn.var_pop('day').cast(IntegerType()).alias('ip_app_channel_var_day'))
train_df = train_df.join(gp, on=['ip', 'app', 'channel'], how='left')
gp = train_df.groupby(['ip', 'app', 'channel']).agg(fn.mean('day').cast(IntegerType()).alias('ip_app_channel_mean_day'))
train_df = train_df.join(gp, on=['ip', 'app', 'channel'], how='left')
gp = train_df.groupby(['ip', 'app', 'channel']).agg(fn.mean('hour').cast(IntegerType()).alias('ip_app_channel_mean_hour'))
train_df = train_df.join(gp, on=['ip', 'app', 'channel'], how='left')
gp = train_df.groupby(['ip', 'app', 'channel']).agg(fn.var_pop('hour').cast(IntegerType()).alias('ip_app_channel_var_hour'))
train_df = train_df.join(gp, on=['ip', 'app', 'channel'], how='left')
gp = train_df.groupby(['ip', 'app', 'os']).agg(fn.mean('hour').cast(IntegerType()).alias('ip_app_os_mean_hour'))
train_df = train_df.join(gp, on=['ip', 'app', 'os'], how='left')
gp = train_df.groupby(['ip', 'app', 'os']).agg(fn.var_pop('hour').cast(IntegerType()).alias('ip_app_os_var_hour'))
train_df = train_df.join(gp, on=['ip', 'app', 'os'], how='left')
#基于ip和device组合的其他特征的unique
gp = train_df.groupby(['ip', 'device']).agg(fn.countDistinct('app').cast(IntegerType()).alias( 'ct4_count'))
train_df= train_df.join(gp, on=['ip', 'device'], how='left')
gp = train_df.groupby(['ip', 'device']).agg(fn.countDistinct('os').cast(IntegerType()).alias( 'ct5_count'))
train_df= train_df.join(gp, on=['ip', 'device'], how='left')
gp = train_df.groupby(['ip', 'device']).agg(fn.countDistinct('channel').cast(IntegerType()).alias( 'ct6_count'))
train_df= train_df.join(gp, on=['ip', 'device'], how='left')
train_data=train_df
#处理测试集
train_df=test_data
#点击时间的拆分    
train_df=train_df.withColumn('hour',fn.hour('click_time').cast(IntegerType()))
train_df=train_df.withColumn('day',fn.dayofmonth('click_time').cast(IntegerType()))
train_df=train_df.withColumn('minute',fn.minute('click_time').cast(IntegerType()))
train_df=train_df.withColumn('second',fn.second('click_time').cast(IntegerType()))
#分组点击次数的衍生特征（按照click_time衍生）
gp=train_df.groupby(['ip', 'app', 'device', 'os']).agg(fn.count('click_time').cast(IntegerType()).alias('ct1_count'))
train_df = train_df.join(gp, on=['ip', 'app', 'device', 'os'], how='left')
gp=train_df.groupby(['ip', 'device', 'os', 'channel']).agg(fn.count('click_time').cast(IntegerType()).alias('ct2_count'))
train_df = train_df.join(gp, on=['ip', 'device', 'os', 'channel'], how='left')
gp=train_df.groupby(['ip', 'app', 'device', 'os', 'channel']).agg(fn.count('click_time').cast(IntegerType()).alias('ct3_count'))
train_df = train_df.join(gp, on=['ip', 'app', 'device', 'os', 'channel'], how='left')
#仅对ip分组，对其他特征做count和unique
gp = train_df.groupby(['ip', 'app']).agg(fn.count('channel').cast(IntegerType()).alias('ip_app_cha_count'))
train_df = train_df.join(gp, on=['ip', 'app'], how='left')
gp = train_df.groupby(['ip','app']).agg(fn.countDistinct('channel').cast(IntegerType()).alias('ip_app_cha_unique'))
train_df = train_df.join(gp, on=['ip','app'], how='left')
gp = train_df.groupby(['ip', 'app', 'os']).agg(fn.count('channel').cast(IntegerType()).alias('ip_app_os_count'))
train_df = train_df.join(gp, on=['ip', 'app', 'os'], how='left')
gp = train_df.groupby(['ip', 'app', 'os']).agg(fn.countDistinct('channel').cast(IntegerType()).alias('ip_app_os_unique'))
train_df = train_df.join(gp, on=['ip', 'app', 'os'], how='left')
gp = train_df.groupby(['ip', 'app', 'channel']).agg(fn.var_pop('day').cast(IntegerType()).alias('ip_app_channel_var_day'))
train_df = train_df.join(gp, on=['ip', 'app', 'channel'], how='left')
gp = train_df.groupby(['ip', 'app', 'channel']).agg(fn.mean('day').cast(IntegerType()).alias('ip_app_channel_mean_day'))
train_df = train_df.join(gp, on=['ip', 'app', 'channel'], how='left')
gp = train_df.groupby(['ip', 'app', 'channel']).agg(fn.mean('hour').cast(IntegerType()).alias('ip_app_channel_mean_hour'))
train_df = train_df.join(gp, on=['ip', 'app', 'channel'], how='left')
gp = train_df.groupby(['ip', 'app', 'channel']).agg(fn.var_pop('hour').cast(IntegerType()).alias('ip_app_channel_var_hour'))
train_df = train_df.join(gp, on=['ip', 'app', 'channel'], how='left')
gp = train_df.groupby(['ip', 'app', 'os']).agg(fn.mean('hour').cast(IntegerType()).alias('ip_app_os_mean_hour'))
train_df = train_df.join(gp, on=['ip', 'app', 'os'], how='left')
gp = train_df.groupby(['ip', 'app', 'os']).agg(fn.var_pop('hour').cast(IntegerType()).alias('ip_app_os_var_hour'))
train_df = train_df.join(gp, on=['ip', 'app', 'os'], how='left')
#基于ip和device组合的其他特征的unique
gp = train_df.groupby(['ip', 'device']).agg(fn.countDistinct('app').cast(IntegerType()).alias( 'ct4_count'))
train_df= train_df.join(gp, on=['ip', 'device'], how='left')
gp = train_df.groupby(['ip', 'device']).agg(fn.countDistinct('os').cast(IntegerType()).alias( 'ct5_count'))
train_df= train_df.join(gp, on=['ip', 'device'], how='left')
gp = train_df.groupby(['ip', 'device']).agg(fn.countDistinct('channel').cast(IntegerType()).alias( 'ct6_count'))
train_df= train_df.join(gp, on=['ip', 'device'], how='left')
test_data=train_df
#训练模型
import mmlspark
from mmlspark import ComputeModelStatistics
from mmlspark import LightGBMClassifier
lgb=LightGBMClassifier(featuresCol=train_data.drop('is_attributed','attributed_time','click_time').columns,baggingFraction=0.7,featureFraction=0.8,
                       labelCol='is_attributed',learningRate=0.01,
                       numIterations=300,numLeaves=30,parallelism ='data_parallel',maxDepth=3,minSumHessianInLeaf=0.001)
lgb_model=lgb.fit(train_data)
predictions=lgb_model.transform(test_data)
metrics = ComputeModelStatistics().transform(predictions)