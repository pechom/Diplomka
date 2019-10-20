from skfeature.function.statistical_based import gini_index
from skfeature.function.similarity_based import SPEC, fisher_score, reliefF, trace_ratio, lap_score
import datetime
import pandas as pd
import numpy as np
import glob
import csv
import os
import shutil
import lightgbm as lgb
import xgboost as xgb


feature_path = 'features/simple.csv'
labels_path = 'subory/clear_labels2_head.csv'
rates = [0.05, 0.1, 0.2, 0.3]
depths = [5, 6, 7, 8]
min_leaves = [5, 10, 15]

min_data_in_bin = [1, 2]
min_data_in_leaf = [5, 10]
max_bin = [1000, 10000, 100000]


def LGBM_goss(data, label, min_bin, min_leaf, bin):
    # https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
    dtrain = lgb.Dataset(data=data, label=label)
    param = {"max_depth": 7, "learning_rate": 0.2, "objective": 'multiclass', 'num_leaves': 80, 'boosting': 'goss',
             "num_class": 10, "metric": 'multi_error', 'min_data_in_leaf': min_leaf, 'num_threads': -1,
             'verbosity': -1, 'max_bin': bin, 'min_data_in_bin': min_bin}
    before = datetime.datetime.now()
    result = lgb.cv(param, dtrain, num_boost_round=100, nfold=10, stratified=True, verbose_eval=None, shuffle=True)
    after = datetime.datetime.now()
    print("LGBM GOSS")
    LGBM_print(result, before, after)


def LGBM(data, label, min_data_in_bin, min_data_in_leaf, max_bin):
    # https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
    dtrain = lgb.Dataset(data=data, label=label)
    param = {"max_depth": 7, "learning_rate": 0.2, "objective": 'multiclass', 'num_leaves': 100,
             "num_class": 10, "metric": 'multi_error', 'min_data_in_leaf': min_data_in_leaf,
             'num_threads': -1, 'verbosity': -1,
             'max_bin': max_bin, 'min_data_in_bin': min_data_in_bin}
    before = datetime.datetime.now()
    result = lgb.cv(param, dtrain, num_boost_round=100, nfold=10, stratified=True, verbose_eval=None, shuffle=True)
    after = datetime.datetime.now()
    print("LGBM")
    LGBM_print(result, before, after)


def LGBM_dart(data, label):
    # https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
    dtrain = lgb.Dataset(data=data, label=label)
    param = {"max_depth": 7, "learning_rate": 0.2, "objective": 'multiclass', 'num_leaves': 80, 'boosting': 'dart',
             "num_class": 10, "metric": 'multi_error', 'min_data_in_leaf': 10, 'num_threads': -1, 'verbosity': -1,
             'min_data_in_bin': 3, 'max_bin': 255, 'xgboost_dart_mode': False}
    before = datetime.datetime.now()
    result = lgb.cv(param, dtrain, num_boost_round=100, nfold=10, stratified=True, verbose_eval=None, shuffle=True)
    after = datetime.datetime.now()
    print("LGBM dart")
    LGBM_print(result, before, after)


def xgboost_dart(data, label):
    # parametre: https://xgboost.readthedocs.io/en/latest/parameter.html
    dtrain = xgb.DMatrix(data, label=label)
    param = {'max_depth': 7, 'objective': 'multi:softmax', 'eval_metric': 'merror', 'num_class': 10,
             'learning_rate': 0.2, 'n_jobs': -1, 'min_child_weight': 10, 'booster': 'dart',
             'rate_drop':0.1, 'skip_drop': 0.5}
    before = datetime.datetime.now()
    result = xgb.cv(param, dtrain, num_boost_round=100, nfold=10, metrics=['merror'], stratified=True, shuffle=True)
    after = datetime.datetime.now()
    print("XGBoost")
    xgb_print(result, before, after)


def xgboost(data, label, min_child_weight):
    # parametre: https://xgboost.readthedocs.io/en/latest/parameter.html
    dtrain = xgb.DMatrix(data, label=label)
    param = {'max_depth': 7, 'objective': 'multi:softmax', 'eval_metric': 'merror', 'num_class': 10,
             'learning_rate': 0.2, 'n_jobs': -1, 'min_child_weight': min_child_weight}
    before = datetime.datetime.now()
    result = xgb.cv(param, dtrain, num_boost_round=100, nfold=10, metrics=['merror'], stratified=True, shuffle=True)
    after = datetime.datetime.now()
    print("XGBoost")
    xgb_print(result, before, after)


def LGBM_print(result, before, after):
    print("najlepsi priemer: " + str(1 - min(result['multi_error-mean'])))
    print("index najlepsieho: " + str(result['multi_error-mean'].index(min(result['multi_error-mean']))))
    print("najhorsi priemer: " + str(1 - max(result['multi_error-mean'])))
    print("finalny priemer: " + str(1 - result['multi_error-mean'][-1]))
    print("cas: " + str(after - before))
    print('\n')


def xgb_print(result, before, after):
    print("najlepsi priemer: " + str(1 - min(result['test-merror-mean'])))
    print("index najlepsieho: " + str(result['test-merror-mean'][result['test-merror-mean'] ==
                                                                 min(result['test-merror-mean'])].index[0]))
    print("najhorsi priemer: " + str(1 - max(result['test-merror-mean'])))
    print("finalny priemer: " + str(1 - result['test-merror-mean'].iloc[-1]))
    print("cas: " + str(after - before))
    print('\n')


labels = np.loadtxt(labels_path, delimiter=',', skiprows=1, dtype=np.uint8)
data = np.loadtxt(feature_path, delimiter=',', skiprows=1, dtype=np.uint64)
print("vsetky data: " + str(len(data[0])))
print('\n')


# for rate in rates:
#     for depth in depths:
#         for leaf in leaves:
#             for min_leaf in min_leaves:
#                 LGBM_goss(data, labels, rate, depth, leaf, min_leaf)

# for min_bin in min_data_in_bin:
#     for min_leaf in min_data_in_leaf:
#         for bin in max_bin:
#             print(str(min_bin) + " " + str(min_leaf) + " " + str(bin))
#             LGBM(data, labels, min_bin, min_leaf, bin)
#
# for min_leaf in min_data_in_leaf:
#     xgboost(data, labels, min_leaf)

# LGBM_goss(data, labels, 3, 10, 255)

LGBM_dart(data, labels)
