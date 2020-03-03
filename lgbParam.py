# -*- coding: utf-8 -*-
# @Time    : 2020-02-21 16:04
# @Author  : HuangSir
# @FileName: lgbParam.py
# @Software: PyCharm
# @Description:lgb参数

from hyperopt import hp
import numpy as np

selectFeatures = 25 # 控制特征数
earlyStopping = 50 # 控制早停
max_evals = 20 # 参数调优次数
select_num_boost_round = 1000 # 特征选择训练轮次
valTime = 20200120 # 划分验证集时间

# 认为删除收费及不稳定变量
drop_col = ['iziScore','ad730Score','dianXinScore','company_province','identity_province','phoneverify']

# 初始特征
selfParam = {
    'objective':'binary', # 二分类,默认是回归
    'boosting':'dart', # 算法类型, gbdt,dart
    'learning_rate':0.01, # 学习率
    'max_depth':6, # 树的最大深度
    'num_leaves':32, # 2**6 = 64
    'max_cat_threshold':10, # 限制类别特征数量
    'min_data_in_leaf':30, # 叶子最小样本
    'feature_fraction':0.7, # 训练特征比例
    'bagging_fraction':0.8, # 训练样本比例 
    'num_threads':8,
    'min_data_in_bin':30, # 单箱数据量
    'max_bin':256, # 最大分箱数 # 超参
    'is_unbalance':True, # 非平衡样本
    'metric':'auc',
    'train_metric':True,
    'verbose':-1,
}

# 超参域
spaceParam = {
    'boosting': hp.choice('boosting',['gbdt','dart']),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.05)),
    'max_bin':hp.quniform('max_bin',100,300,20), # max_bin
    'num_leaves': hp.quniform('num_leaves', 3, 63, 3), # 较小的num_leaves
    'feature_fraction': hp.uniform('feature_fraction', 0.7,1), # 较小的
    'min_data_in_leaf': hp.quniform('min_data_in_leaf', 10, 50,5), # 较大的
    'num_boost_round':hp.quniform('num_boost_round',500,2000,100), # 迭代次数
    'bagging_fraction':hp.uniform('bagging_fraction',0.6,1)  # 较小的
}