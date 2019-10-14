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


feature_path = 'features/simple.csv'
labels_path = 'subory/clear_labels2_head.csv'
rates = [0.05, 0.1, 0.2, 0.3]
depths = [5, 6, 7, 8]
leaves = [70, 80, 90]
min_leaves = [5, 10, 15]


def LGBM_goss(data, label, learning_rate, max_depth, num_leaves, min_leaf):
    # https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
    dtrain = lgb.Dataset(data=data, label=label)
    param = {"max_depth": 7, "learning_rate": 0.2, "objective": 'multiclass', 'num_leaves': 80, 'boosting': 'goss',
             "num_class": 10, "metric": 'multi_error', 'min_data_in_leaf': 10, 'num_threads': -1, 'verbosity': -1}
    before = datetime.datetime.now()
    result = lgb.cv(param, dtrain, num_boost_round=100, nfold=10, stratified=True, verbose_eval=None, shuffle=True)
    after = datetime.datetime.now()
    print("LGBM GOSS")
    LGBM_print(result, before, after)


def LGBM_print(result, before, after):
    print("najlepsi priemer: " + str(1 - min(result['multi_error-mean'])))
    print("index najlepsieho: " + str(result['multi_error-mean'].index(min(result['multi_error-mean']))))
    print("najhorsi priemer: " + str(1 - max(result['multi_error-mean'])))
    print("finalny priemer: " + str(1 - result['multi_error-mean'][-1]))
    print("cas: " + str(after - before))
    print('\n')


labels = np.loadtxt(labels_path, delimiter=',', skiprows=1, dtype=np.uint8)
data = np.loadtxt(feature_path, delimiter=',', skiprows=1, dtype=np.uint64)
print("vsetky data: " + str(len(data[0])))
print('\n')


for rate in rates:
    for depth in depths:
        for leaf in leaves:
            for min_leaf in min_leaves:
                LGBM_goss(data, labels, rate, depth, leaf, min_leaf)
