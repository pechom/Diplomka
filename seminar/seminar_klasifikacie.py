import numpy as np
import lightgbm as lgb
import xgboost as xgb
import datetime
import sys
import glob
import os

feature_path = 'seminar/simple_discrete.csv'
standard_feature_path = 'seminar/simple_standard.csv'
labels_path = 'seminar/clear_labels2_head.csv'
selected_dir = 'seminar/selection/*'
sys.stdout = open('seminar/classification_times.txt', 'w')


def LGBM(data, label):
    # https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
    dtrain = lgb.Dataset(data=data, label=label)
    param = {"max_depth": 8, "learning_rate": 0.2, "objective": 'multiclass', 'num_leaves': 80,
             "num_class": 10, "metric": 'multi_error', 'min_data_in_leaf': 10, 'num_threads': -1}
    before = datetime.datetime.now()
    result = lgb.cv(param, dtrain, num_boost_round=60, nfold=10, stratified=True, verbose_eval=None, shuffle=True)
    after = datetime.datetime.now()
    print("LGBM")
    print("najlepsi priemer: " + str(1 - min(result['multi_error-mean'])))
    print("index najlepsieho: " + str(result['multi_error-mean'].index(min(result['multi_error-mean']))))
    print("najhorsi priemer: " + str(1 - max(result['multi_error-mean'])))
    print("finalny priemer: " + str(1 - result['multi_error-mean'][-1]))
    print("cas: " + str(after - before))
    print('\n')


def xgboost(data, label):
    # parametre: https://xgboost.readthedocs.io/en/latest/parameter.html
    dtrain = xgb.DMatrix(data, label=label)
    param = {'max_depth': 8, 'objective': 'multi:softmax', 'eval_metric': 'merror', 'num_class': 10,
             'learning_rate': 0.2, 'n_jobs': -1, 'tree_method': 'hist'}
    before = datetime.datetime.now()
    result = xgb.cv(param, dtrain, num_boost_round=60, nfold=10, metrics=['merror'], stratified=True, shuffle=True)
    after = datetime.datetime.now()
    print("XGBoost")
    print("najlepsi priemer: " + str(1 - min(result['test-merror-mean'])))
    print("index najlepsieho: " + str(result['test-merror-mean'][result['test-merror-mean'] ==
                                                                 min(result['test-merror-mean'])].index[0]))
    print("najhorsi priemer: " + str(1 - max(result['test-merror-mean'])))
    print("finalny priemer: " + str(1 - result['test-merror-mean'].iloc[-1]))
    print("cas: " + str(after - before))
    print('\n')


labels = np.loadtxt(labels_path, delimiter=',', skiprows=1, dtype=np.int8)
data = np.loadtxt(feature_path, delimiter=',', skiprows=1, dtype=np.float)
print("vsetky data: " + str(len(data[0])))
xgboost(data, labels)
LGBM(data, labels)
files = glob.glob(selected_dir)
for file in files:
    data = np.loadtxt(file, delimiter=',', skiprows=1, dtype=np.float)
    print(os.path.basename(file)[:-4] + ": " + str(len(data[0])))
    LGBM(data, labels)
sys.stdout.close()
