import numpy as np
import lightgbm as lgb
import datetime
import sys
import glob
import os

feature_path = 'seminar/features.csv'
standard_feature_path = 'seminar/simple_standard.csv'
labels_path = 'seminar/clear_labels2_head.csv'
selected_dir = 'seminar/selection/*'
sys.stdout = open('seminar/classification_times.txt', 'w')


def LGBM(data, label):
    # https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
    dtrain = lgb.Dataset(data=data, label=label)
    param = {"max_depth": 5, "learning_rate": 0.2, "objective": 'multiclass', 'num_leaves': 80,
             "num_class": 10, "metric": 'multi_error', 'min_data_in_leaf': 10, 'num_threads': -1}
    before = datetime.datetime.now()
    result = lgb.cv(param, dtrain, num_boost_round=5, nfold=2, stratified=True, verbose_eval=None, shuffle=True)
    after = datetime.datetime.now()
    print("LGBM")
    print("najlepsi priemer: " + str(1 - min(result['multi_error-mean'])))
    print("najhorsi priemer: " + str(1 - max(result['multi_error-mean'])))
    print("cas: " + str(after - before))
    print('\n')


labels = np.loadtxt(labels_path, delimiter=',', skiprows=1, dtype=np.int8)
data = np.loadtxt(feature_path, delimiter=',', skiprows=1, dtype=np.float)
files = glob.glob(selected_dir)
for file in files:
    data = np.loadtxt(file, delimiter=',', skiprows=1, dtype=np.float)
    print(os.path.basename(file)[:-4])
    LGBM(data, labels)
sys.stdout.close()
