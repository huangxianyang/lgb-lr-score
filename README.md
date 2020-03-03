# lgb-lr-score
lightgbm+lr评分模型, 并基于贝叶斯调参方法.
```python
# -*- coding: utf-8 -*-
# @Time    : 2020-02-21 15:46
# @Author  : HuangSir
# @FileName: lgb_lr.py
# @Software: PyCharm
# @Description:lgb+lr原生接口模型模型,特征选择及模型训练
import json
import pandas as pd
import numpy as np
import riskModel as rs
import lightgbm as lgb
from hyperopt import STATUS_OK,tpe,fmin,Trials
from apkModel.lgbParam import *
import matplotlib.pyplot as plt
import joblib

def getParam(param):
    for k in ['max_bin', 'num_leaves', 'min_data_in_leaf','num_boost_round']:
        param[k] = int(float(param[k]))
    for k in ['learning_rate', 'feature_fraction','bagging_fraction']:
        param[k] = float(param[k])
    if param['boosting'] == 0:
        param['boosting'] = 'gbdt'
    elif param['boosting'] == 1:
        param['boosting'] = 'dart'
    # 添加固定参数
    param['objective'] = 'binary'
    param['max_depth'] = 7
    param['num_threads'] = 8
    param['is_unbalance'] = True
    param['metric'] = 'auc'
    param['train_metric'] = True
    param['verbose'] = -1
    return param

def lossFun(param):
    param = getParam(param)
    m = lgb.train(params=param,train_set=train_data,num_boost_round=param['num_boost_round'],
                          valid_sets=[train_data,valid_data],valid_names=['train','valid'],
                          feature_name=features,categorical_feature=cat_feature,
                          early_stopping_rounds=earlyStopping,verbose_eval=False,keep_training_booster=True)
    train_auc = m.best_score['train']['auc']
    valid_auc = m.best_score['valid']['auc']
    loss_auc = train_auc - valid_auc
    print('训练集auc:{},测试集auc:{},loss_auc:{}'.format(train_auc, valid_auc, loss_auc))
    return {'loss': loss_auc, 'params': param, 'status': STATUS_OK}

if __name__ == '__main__':
    path = './lgb-lr/' # 结果存储
    data = pd.read_excel('./data/dt.xlsx', sheet_name='data')#.sample(1000)
    varDict = pd.read_excel('./data/dt.xlsx', sheet_name='vardict')
    # 特征类型
    cat_feature = varDict.loc[varDict.type == 'category', 'variable'].tolist()
    num_feature = varDict.loc[varDict.type == 'numerical', 'variable'].tolist()

    # 移除收费特征及不稳定特征
    for col in drop_col:
        try:
            num_feature.remove(col)
        except:
            try:
                cat_feature.remove(col)
            except:
                pass

    cat_feature.append('old_customer') # 加入新老客特征
    features = cat_feature + num_feature
    print('原始特征集:{},{}'.format(len(features), features))

    # 标记验证样本 2020年01月15日后为验证集
    data['isValid'] = data.apply(lambda df: 1 if int(str(df['loanId'])[-8:]) >= valTime else 0, axis=1)
    print('样本划分分布:', data['isValid'].value_counts())

    ##################################
    # 类别变量数值化(label编码)
    m  = rs.Preprocess()
    data,category_map = m.factor_map(df=data,col_list=cat_feature)
    joblib.dump(category_map,path+'category_map.pkl')
    #################################
    # 样本划分
    train = data.loc[data['isValid']==0,:] # 训练集
    valid  = data.loc[data['isValid']==1,:] # 验证集

    # 特征选择 ---------------------------------------------------------------------------------
    train_data = lgb.Dataset(data=train[features],label=train['target'],feature_name=features,categorical_feature=cat_feature)
    valid_data = lgb.Dataset(data=valid[features],label=valid['target'],reference=train_data,feature_name=features,categorical_feature=cat_feature)

    sm = lgb.train(params=selfParam,train_set=train_data,num_boost_round=select_num_boost_round,
                          valid_sets=[train_data,valid_data],valid_names=['train','valid'],
                          feature_name=features,categorical_feature=cat_feature,
                          early_stopping_rounds=earlyStopping,verbose_eval=False,keep_training_booster=True)
    features_importance = {k:v for k,v in zip(sm.feature_name(),sm.feature_importance(iteration=sm.best_iteration))}
    sort_feature_importance = sorted(features_importance.items(),key=lambda x:x[1],reverse=True)
    print('total feature best score:', sm.best_score)
    print('total feature importance:',sort_feature_importance)
    print('select forward {} features:{}'.format(selectFeatures,sort_feature_importance[:selectFeatures]))
    model_feature = [k[0] for k in sort_feature_importance[:selectFeatures]]

    # 参数调优 -------------------------------------------------------------------------------
    cat_feature = list(set(cat_feature+['old_customer'])&set(model_feature)) # 新老客特征必须作为入参
    num_feature = list(set(num_feature)&set(model_feature))
    features = cat_feature+num_feature
    train_data = lgb.Dataset(data=train[features],label=train['target'],feature_name=features,categorical_feature=cat_feature)
    valid_data = lgb.Dataset(data=valid[features],label=valid['target'],reference=train_data,feature_name=features,categorical_feature=cat_feature)

    best_param = fmin(fn=lossFun, space=spaceParam, algo=tpe.suggest, max_evals=max_evals, trials=Trials())
    best_param = getParam(best_param)
    print('Search best param:',best_param)

    # 模型训练 --------------------------------------------------------------------------------
    evals_result = {}
    lgbModel = lgb.train(params=best_param,train_set=train_data,num_boost_round=best_param['num_boost_round'],
                          valid_sets=[train_data,valid_data],valid_names=['train','valid'],evals_result=evals_result,
                          feature_name=features,categorical_feature=cat_feature,
                          early_stopping_rounds=earlyStopping,verbose_eval=False,keep_training_booster=True)

    features_importance = {k:v for k,v in zip(lgbModel.feature_name(),lgbModel.feature_importance(iteration=lgbModel.best_iteration))}
    sort_feature_importance = sorted(features_importance.items(),key=lambda x:x[1],reverse=True)
    print('Last model best score:', lgbModel.best_score)
    print('Last model feature importance:',sort_feature_importance)

    # 保存模型
    lgbModel.save_model(path + 'lgbModel.txt', num_iteration=lgbModel.best_iteration)
    _ = lgbModel.dump_model(num_iteration=lgbModel.best_iteration)
    with open(path+'lgbModel.json','w',encoding='utf-8') as f:
        json.dump(_,f,ensure_ascii=False)

    # 特征重要度
    _ = lgb.plot_importance(booster=lgbModel,max_num_features=selectFeatures,figsize=(15,8))
    plt.savefig(path+'feature_importance.png')
    plt.close()
    # 树模型图
    graph = lgb.create_tree_digraph(lgbModel, tree_index=5, name='Tree5')
    graph.render(filename=path+'Tree5')

    # 评分卡校准 --------------------------------------------------------------------------------
    # lgb预测
    train['lgb_prob'] = lgbModel.predict(train[features],num_iteration=lgbModel.best_iteration)
    valid['lgb_prob']  = lgbModel.predict(valid[features],num_iteration=lgbModel.best_iteration)
    print('LGB train:', rs.model_norm(train['target'],train['lgb_prob']))
    print('LGB valid:', rs.model_norm(valid['target'],valid['lgb_prob']))
    # 拟合LR,已达到评分卡校准
    lr = rs.TrainLr(df_woe=train, features=['lgb_prob'], target='target',class_weight='balanced')
    lr = lr.lr(C=1.0,filename=path + 'train_')
    train['lr_prob'] = lr.predict_proba(train['lgb_prob'].values.reshape(-1, 1))[:, 1]
    valid['lr_prob'] = lr.predict_proba(valid['lgb_prob'].values.reshape(-1, 1))[:, 1]
    print('LR train:', rs.model_norm(train['target'], train['lr_prob']))
    print('LR valid:', rs.model_norm(valid['target'], valid['lr_prob']))
    # 作图
    mplt = rs.PlotModel(y_true=valid['target'].values, y_prob=valid['lr_prob'].values)
    mplt.plot_roc_curve(filename=path + 'valid_')
    mplt.plot_ks_curve(filename=path + 'valid_')

    valid['score'] = rs.Prob2Score(prob=valid['lr_prob'], basePoint=600, PDO=50)  # 分数映射
    train['score'] = rs.Prob2Score(prob=train['lr_prob'], basePoint=600, PDO=50)  # 分数映射
    # 保存模型
    joblib.dump(lgbModel,path+'lgbModel.pkl')
    joblib.dump(lr,path+'lrModel.pkl')

    # 输出模型报告 --------------------------------------------------------------------------------
    report = rs.stragety_score(score_df=valid, step=10, score='score', label='target')
    del report['profit']
    report.to_excel(path + 'ModelReport.xlsx')
```
