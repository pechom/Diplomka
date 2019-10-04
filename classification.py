import xgboost as xgb
import pandas as pd
import numpy as np
import lightgbm as lgb
import datetime
from rgf.sklearn import RGFClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
import catboost as cat
import sys
import glob
import os
import preprocessing

# feature_path = 'seminar/features.csv'
feature_path = 'C:/PycharmProjects/Diplomka/skusobny/classification/sections.csv'
standard_feature_path = 'C:/PycharmProjects/Diplomka/skusobny/classification/standard_sections.csv'
labels_path = 'seminar/clear_labels2_head.csv'
selected_dir = 'seminar/selection/*'  # kde sa ulozili skupiny atributov po selekcii
standard_selected_dir = 'seminar/standard/'
sys.stdout = open('seminar/classification_times.txt', 'w')


def panda_load():
    labels = pd.read_csv(labels_path, dtype=np.int8)  # pri sklearn treba mat label.values.ravel()
    standard_data = pd.read_csv(standard_feature_path, dtype=np.float)
    data = pd.read_csv(feature_path)
    return data, standard_data, labels.values.ravel()


def numpy_load(labels_path, feature_path):
    labels = np.loadtxt(labels_path, delimiter=',', skiprows=1, dtype=np.int8)
    data = np.loadtxt(feature_path, delimiter=',', skiprows=1, dtype=np.uint32)
    return data, labels


def numpy_standard_load():
    labels = np.loadtxt(labels_path, delimiter=',', skiprows=1, dtype=np.int8)
    standard_data = np.loadtxt(standard_feature_path, delimiter=',', skiprows=1, dtype=np.float)
    return standard_data, labels


def xgboost(data, label):
    # parametre: https://xgboost.readthedocs.io/en/latest/parameter.html
    dtrain = xgb.DMatrix(data, label=label)
    param = {'max_depth': 7, 'objective': 'multi:softmax', 'eval_metric': 'merror', 'num_class': 10,
             'learning_rate': 0.2, 'n_jobs': -1}
    before = datetime.datetime.now()
    result = xgb.cv(param, dtrain, num_boost_round=20, nfold=10, metrics=['merror'], stratified=True, shuffle=True)
    after = datetime.datetime.now()
    print("XGBoost")
    xgb_print(result, before, after)


def hist_xgboost(data, label):
    dtrain = xgb.DMatrix(data, label=label)
    param = {'max_depth': 7, 'objective': 'multi:softmax', 'eval_metric': 'merror', 'num_class': 10,
             'learning_rate': 0.2, 'n_jobs': -1, 'tree_method': 'hist'}
    before = datetime.datetime.now()
    result = xgb.cv(param, dtrain, num_boost_round=20, nfold=10, metrics=['merror'], stratified=True, shuffle=True)
    after = datetime.datetime.now()
    print("XGBoost hist")
    xgb_print(result, before, after)


def linear_xgb(data, label):  # implicitna selekcia atributov
    dtrain = xgb.DMatrix(data, label=label)
    param = {'max_depth': 7, 'objective': 'multi:softmax', 'eval_metric': 'merror', 'num_class': 10,
             'learning_rate': 0.2, 'n_jobs': -1, 'booster': 'gblinear',
             'feature_selector': 'thrifty', 'updater': 'coord_descent'}
    before = datetime.datetime.now()
    cvresult = xgb.cv(param, dtrain, num_boost_round=20, nfold=10, metrics=['merror'], stratified=True)
    after = datetime.datetime.now()
    print("linear XGBoost")
    xgb_print(cvresult, before, after)


def xgb_print(result, before, after):
    print("najlepsi priemer: " + str(1 - min(result['test-merror-mean'])))
    print("index najlepsieho: " + str(result['test-merror-mean'][result['test-merror-mean'] ==
                                                                 min(result['test-merror-mean'])].index[0]))
    print("najhorsi priemer: " + str(1 - max(result['test-merror-mean'])))
    print("finalny priemer: " + str(1 - result['test-merror-mean'].iloc[-1]))
    print("cas: " + str(after - before))
    print('\n')


def LGBM(data, label):
    # https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
    dtrain = lgb.Dataset(data=data, label=label)
    param = {"max_depth": 7, "learning_rate": 0.2, "objective": 'multiclass', 'num_leaves': 80,
             "num_class": 10, "metric": 'multi_error', 'min_data_in_leaf': 10, 'num_threads': -1}
    before = datetime.datetime.now()
    result = lgb.cv(param, dtrain, num_boost_round=20, nfold=10, stratified=True, verbose_eval=None, shuffle=True)
    after = datetime.datetime.now()
    print("LGBM")
    print("najlepsi priemer: " + str(1 - min(result['multi_error-mean'])))
    print("index najlepsieho: " + str(result['multi_error-mean'].index(min(result['multi_error-mean']))))
    print("najhorsi priemer: " + str(1 - max(result['multi_error-mean'])))
    print("finalny priemer: " + str(1 - result['multi_error-mean'][-1]))
    print("cas: " + str(after - before))
    print('\n')


def CAT(data, label):
    # velmi pomaly, ale najlepsie vysledky
    pool = cat.Pool(data, label, has_header=False)
    params = {
        "loss_function": 'MultiClassOneVsAll', "eval_metric": 'MultiClassOneVsAll', "max_depth": 7,
        "learning_rate": 0.2,  "classes_count": 10, "task_type": 'CPU', "thread_count": 4, "verbose_eval": False}
    before = datetime.datetime.now()
    results = cat.cv(pool=pool, params=params, num_boost_round=20, fold_count=10, shuffle=True, stratified=True,
                     verbose=False)
    after = datetime.datetime.now()
    print("CatBoost")
    print("najlepsi priemer: " + str(1 - min(results['test-MultiClassOneVsAll-mean'])))
    print("index najlepsieho: " + str(results['test-MultiClassOneVsAll-mean'][results['test-MultiClassOneVsAll-mean'] ==
                                                                min(results['test-MultiClassOneVsAll-mean'])].index[0]))
    print("najhorsi priemer: " + str(1 - max(results['test-MultiClassOneVsAll-mean'])))
    print("finalny priemer: " + str(1 - results['test-MultiClassOneVsAll-mean'].iloc[-1]))
    print("cas: " + str(after - before))
    print('\n')


def multi_cv_fit(model, n_splits, n_repeats, data, label):
    before = datetime.datetime.now()
    scores = cross_val_score(model, data, label, scoring='accuracy',
                             cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1))
    after = datetime.datetime.now()
    means = [sum(scores[i:i + n_splits]) / n_splits for i in range(0, len(scores), n_splits)]
    print("najlepsi priemer: " + str(max(means)))
    print("najhorsi priemer: " + str(min(means)))
    print("cas: " + str(after - before))
    print('\n')


def cv_fit(model, n_splits, data, label):
    before = datetime.datetime.now()
    scores = cross_val_score(model, data, label, scoring='accuracy',
                             cv=StratifiedKFold(n_splits=n_splits, random_state=1))
    after = datetime.datetime.now()
    print("najlepsi vysledok: " + str(max(scores)))
    print("najhorsi vysledok: " + str(min(scores)))
    print(scores)
    print("cas: " + str(after - before))
    print('\n')


# velmi pomaly, slabsie vysledky
def RGF(data, label):
    rgf = RGFClassifier(max_leaf=80, algorithm="RGF_Sib", verbose=False, n_jobs=-1, min_samples_leaf=10,
                        learning_rate=0.2)
    # multi_cv_fit(rgf, 5, 20)
    print("RGF")
    cv_fit(rgf, 10, data, label)


def RFC(data, label):
    rfc = RandomForestClassifier(n_estimators=1000, max_depth=7, min_samples_leaf=10, max_leaf_nodes=100,
                                 bootstrap=True, n_jobs=-1, warm_start=False)
    print("RFC")
    cv_fit(rfc, 10, data, label)


def SGD(data, label):
    sgd = SGDClassifier(
        loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, max_iter=1000, tol=0.001, shuffle=True, verbose=0,
        n_jobs=-1, learning_rate='optimal', eta0=0.0, power_t=0.5)
    print("linear SGD SVC")
    # multi_cv_fit(sgd, 5, 20, data, label)
    cv_fit(sgd, 10, data, label)


def LSVC(data, label):
    svc = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=0.001, C=1, multi_class='ovr',
                    fit_intercept=True, verbose=0, max_iter=1000)
    # ak mam standardizovane data tak fit_intercept mozem dat na False
    # ak mam vela atributov tak dam dual na True
    print("linear SVC [liblinear]")
    # multi_cv_fit(svc, 5, 20, data, label)
    cv_fit(svc, 10, data, label)


def SVM(data, label):
    svm = SVC(C=1.0, kernel='rbf', shrinking=True, probability=False, tol=0.001, cache_size=200, verbose=False,
              max_iter=1000, decision_function_shape='ovr', gamma='scale')
    print("RBF SVC")
    # multi_cv_fit(svm, 5, 20, data, label)
    cv_fit(svm, 10, data, label)
    svm = SVC(C=1.0, kernel='linear', shrinking=True, probability=False, tol=0.001, cache_size=200, verbose=False,
              max_iter=1000, decision_function_shape='ovr')
    print("linear SVC [libsvm]")
    # multi_cv_fit(svm, 5, 20, data, label)
    cv_fit(svm, 10, data, label)
    # maju velmi zle vysledky, aj na normalizovanych datach:
    svm = SVC(C=1.0, kernel='poly', shrinking=True, probability=False, tol=0.001, cache_size=200, verbose=False,
              max_iter=1000, decision_function_shape='ovr', gamma='scale')
    print("polynom SVC")
    # multi_cv_fit(svm, 5, 20, data, label)
    cv_fit(svm, 10, data, label)
    svm = SVC(C=1.0, kernel='sigmoid', shrinking=True, probability=False, tol=0.001, cache_size=200, verbose=False,
              max_iter=1000, decision_function_shape='ovr', gamma='scale')
    print("sigmoid SVC")
    # multi_cv_fit(svm, 5, 20, data, label)
    cv_fit(svm, 5, data, label)


labels = np.loadtxt(labels_path, delimiter=',', skiprows=1, dtype=np.int8)
data = np.loadtxt(feature_path, delimiter=',', skiprows=1, dtype=np.uint32)
print("vsetky data: " + str(len(data[0])))
print('\n')
xgboost(data, labels)
hist_xgboost(data, labels)
linear_xgb(data, labels)
LGBM(data, labels)
CAT(data, labels)
RGF(data, labels)
RFC(data, labels)

labels = np.loadtxt(labels_path, delimiter=',', skiprows=1, dtype=np.int8)
standard_data = np.loadtxt(standard_feature_path, delimiter=',', skiprows=1, dtype=np.float)
LSVC(standard_data, labels)
SGD(standard_data, labels)
SVM(standard_data, labels)

files = glob.glob(selected_dir)
labels = np.loadtxt(labels_path, delimiter=',', skiprows=1, dtype=np.int8)
for file in files:
    data = np.loadtxt(file, delimiter=',', skiprows=1, dtype=np.uint32)
    print(os.path.basename(file)[:-4] + ": " + str(len(data[0])))
    xgboost(data, labels)
    hist_xgboost(data, labels)
    linear_xgb(data, labels)
    LGBM(data, labels)
    CAT(data, labels)
    RGF(data, labels)
    RFC(data, labels)
    print("------------------------------------------------------------")
    print('\n')

preprocessing.standardize(selected_dir, standard_selected_dir)
labels = np.loadtxt(labels_path, delimiter=',', skiprows=1, dtype=np.int8)
files = glob.glob(standard_selected_dir+'*')
for file in files:
    standard_data = np.loadtxt(file, delimiter=',', skiprows=1, dtype=np.float)
    print(os.path.basename(file)[:-4] + ": " + str(len(data[0])))
    LSVC(standard_data, labels)
    SGD(standard_data, labels)
    SVM(standard_data, labels)
    print("------------------------------------------------------------")
    print('\n')

sys.stdout.close()
