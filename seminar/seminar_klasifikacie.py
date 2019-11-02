import numpy as np
import lightgbm as lgb
import datetime
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
import sys
import statistics
import glob
import os
from sklearn.preprocessing import StandardScaler
import csv

feature_path = 'data.csv'
standard_feature_path = 'standard_data.csv'
labels_path = 'labels.csv'
selected_dir = 'selection/*'
standard_selected_dir = 'selection_standard/'


def LGBM_goss(data, label):
    # https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
    dtrain = lgb.Dataset(data=data, label=label)
    param = {"max_depth": 7, "learning_rate": 0.2, "objective": 'multiclass', 'num_leaves': 80, 'boosting': 'goss',
             "num_class": 10, "metric": 'multi_error', 'min_data_in_leaf': 10, 'num_threads': -1, 'verbosity': -1,
             'min_data_in_bin': 3, 'max_bin': 255, 'enable_bundle': True, 'max_conflict_rate': 0.0}
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


def SVM(data, label, kernel, message):
    svm = SVC(C=1.0, kernel=kernel, shrinking=True, probability=False, tol=0.001, cache_size=200, verbose=False,
              max_iter=-1, decision_function_shape='ovr', gamma='scale')
    print(message)
    cv_fit(svm, 10, data, label)


def cv_fit(model, n_splits, data, label):
    before = datetime.datetime.now()
    scores = cross_val_score(model, data, label, scoring='accuracy',
                             cv=StratifiedKFold(n_splits=n_splits, random_state=1))
    after = datetime.datetime.now()
    print("priemerny vysledok: " + str(statistics.mean(scores)))
    print("cas: " + str(after - before))
    print('\n')


def standardize(input_path, standard_path):
    files = glob.glob(input_path)
    for name in files:
        if os.path.isfile(name):
            with open(name) as f:
                reader = csv.reader(f, delimiter=',')
                header = next(reader)
                data = list(reader)
            with open(standard_path + os.path.basename(f.name), "w", newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow(header)
                scaler = StandardScaler(copy=True)
                standard_data = scaler.fit_transform(data)
                writer.writerows(standard_data)


labels = np.loadtxt(labels_path, delimiter=',', skiprows=1, dtype=np.int8)
data = np.loadtxt(feature_path, delimiter=',', skiprows=1, dtype=np.float)
sys.stdout = open('classification_times.txt', 'w')
print("vsetky data: " + str(len(data[0])))
print('\n')
LGBM_goss(data, labels)
standard_data = np.loadtxt(standard_feature_path, delimiter=',', skiprows=1, dtype=np.float64)
SVM(standard_data, labels, 'linear', "linear SVC [libsvm]")
sys.stdout.close()
sys.stdout = open('classification_selected.txt', 'w')
files = glob.glob(selected_dir)
for file in files:
    data = np.loadtxt(file, delimiter=',', skiprows=1, dtype=np.uint64)
    print(os.path.basename(file)[:-4] + ": " + str(len(data[0])))
    LGBM_goss(data, labels)
    print("------------------------------------------------------------")
    print('\n')
files = glob.glob(standard_selected_dir + '*')
for file in files:
    data = np.loadtxt(file, delimiter=',', skiprows=1, dtype=np.float64)
    print(os.path.basename(file)[:-4] + ": " + str(len(data[0])))
    LGBM_goss(data, labels)
    print("------------------------------------------------------------")
    print('\n')
standardize(selected_dir, standard_selected_dir)
files = glob.glob(standard_selected_dir + '*')
for file in files:
    standard_data = np.loadtxt(file, delimiter=',', skiprows=1, dtype=np.float64)
    print(os.path.basename(file)[:-4] + ": " + str(len(standard_data[0])))
    SVM(standard_data, labels, 'linear', "linear SVC [libsvm]")
    print("------------------------------------------------------------")
    print('\n')
for file in files:
    if not (os.path.basename(file).startswith("SVC") or os.path.basename(file).startswith("SVM") or
            os.path.basename(file).startswith("SGD")):
        os.remove(file)
sys.stdout.close()
