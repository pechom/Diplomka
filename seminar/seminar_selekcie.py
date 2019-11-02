import numpy as np
import pandas as pd
import csv
from sklearn.svm import LinearSVC
from pyHSICLasso import HSICLasso
import lightgbm as lgb
from sklearn.feature_selection import SelectFromModel
import sys
import datetime

feature_path = 'data.csv'
standard_feature_path = 'standard_data.csv'
labels_path = 'labels.csv'
output_dir = 'selection/'
sys.stdout = open('selection_times.txt', 'w')
np.set_printoptions(threshold=np.inf)


def save_to_csv(transformed_data, selected, prefix):
    with open(output_dir + prefix + ".csv", "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(header[selected])
        writer.writerows(transformed_data)


def transform_and_save(selected, prefix):
    # transformed_data = np.delete(data, not_selected, axis=1)  # vymazem nevybrane stlpce
    with open(output_dir + prefix + ".csv", "w", newline='') as csv_file:  # zapisem len vybrane
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(header[selected])
        writer.writerows(data[:, selected])


def model_selection_treshold(model, prefix):
    selection = SelectFromModel(model, threshold=-np.inf, prefit=True, max_features=treshold)
    new_data = selection.transform(data)
    selected = selection.get_support(True)
    print(len(selected))
    if len(selected) < len(header):
        save_to_csv(new_data, selected, prefix)


def model_fit(model, prefix):
    before = datetime.datetime.now()
    model.fit(data, labels)
    after = datetime.datetime.now()
    model_selection_treshold(model, prefix)
    print("cas: " + str(after - before))
    print('\n')


def LGBM():
    model_split = lgb.LGBMClassifier(max_depth=7, learning_rate=0.2, n_estimators=100, objective='multiclass',
                                     n_jobs=-1, num_leaves=80, min_child_samples=10, importance_type='split',  num_class=10)
    model_gain = lgb.LGBMClassifier(max_depth=7, learning_rate=0.2, n_estimators=100, objective='multiclass',
                                    n_jobs=-1, num_leaves=80, min_child_samples=10, importance_type='gain',  num_class=10)
    print("LGBM split")
    model_fit(model_split, "LGBM split")
    print("LGBM gain")
    model_fit(model_gain, "LGBM gain")


def LSVC_l1():
    model_l1 = LinearSVC(penalty='l1', loss='squared_hinge', dual=False, tol=0.001, C=1, multi_class='ovr',
                         fit_intercept=False, verbose=0, max_iter=1000)
    # ak mam standardizovane data tak fit_intercept mozem dat na False
    # ak mam vela atributov tak dam dual na True
    print("SVC L1")
    model_fit(model_l1, "SVC_L1")


def HSIC_lasso(treshold):
    hsic = HSICLasso()
    hsic.input(data, labels)
    before = datetime.datetime.now()
    hsic.classification(treshold, B=17, M=3)
    # B a M su na postupne nacitanie ak mam velky dataset, B deli pocet vzoriek, pre klasicky algoritmus B=0, M=1
    after = datetime.datetime.now()
    print("HSIC Lasso")
    selected = hsic.get_index()
    print(len(selected))
    print("cas: " + str(after - before))
    print('\n')
    if len(selected) < len(header):
        transform_and_save(selected, "HSIC_Lasso")


labels = np.loadtxt(labels_path, delimiter=',', skiprows=1, dtype=np.uint8)
header = pd.read_csv(feature_path, nrows=1, header=None)
header = header.to_numpy()[0]
data = np.loadtxt(feature_path, delimiter=',', skiprows=1, dtype=np.uint64)
print("pocet atributov: " + str(len(header)))
print('\n')
treshold = 80

LGBM()
HSIC_lasso(treshold)
output_dir = 'selection_standard/'
data = np.loadtxt(standard_feature_path, delimiter=',', skiprows=1, dtype=np.float64)
LSVC_l1()