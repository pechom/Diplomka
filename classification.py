import xgboost as xgb
import pandas as pd
import numpy as np
import lightgbm as lgb
import datetime
from rgf.sklearn import RGFClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
import catboost as cat
import sys
import statistics
import glob
import os
import preprocessing
import warnings
import pickle
import labels
import csv

feature_file = 'features/original.csv'
standard_feature_file = 'features/standard/original.csv'
labels_path = 'subory/labels.csv'
cluster_labels_outliers_path = 'subory/cluster_labels.txt'
selected_dir = 'features/selection/*'  # kde sa ulozili skupiny atributov po selekcii
standard_selected_dir = 'features/selection_standard/'
results_path = 'results/'
classification_selected_dir = results_path + 'selected/'
times_path = results_path + 'classification_times/'
trained_path = 'subory/trained/'
predictions_file = classification_selected_dir + 'predictions_selected.csv'
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
mode = "prediction"
labels_for_predict = labels.consensus_labels_file

multi_cv_repeats = 5
boost_rounds = 100
max_iter = 1000
num_class = 11
cv_fold = 10


def panda_load():
    labels = pd.read_csv(labels_path, dtype=np.int8)  # pri sklearn treba mat label.values.ravel()
    standard_data = pd.read_csv(standard_feature_file, dtype=np.float)
    data = pd.read_csv(feature_file)
    return data, standard_data, labels.values.ravel()


def numpy_load(labels_path, feature_path):
    labels = np.loadtxt(labels_path, delimiter=',', skiprows=1, dtype=np.uint8)
    data = np.loadtxt(feature_path, delimiter=',', skiprows=1, dtype=np.uint64)
    return data, labels


def numpy_standard_load():
    labels = np.loadtxt(labels_path, delimiter=',', skiprows=1, dtype=np.int8)
    standard_data = np.loadtxt(standard_feature_file, delimiter=',', skiprows=1, dtype=np.float)
    return standard_data, labels


def xgboost(data, label):
    # parametre: https://xgboost.readthedocs.io/en/latest/parameter.html
    dtrain = xgb.DMatrix(data, label=label)
    param = {'max_depth': 7, 'objective': 'multi:softmax', 'eval_metric': 'merror', 'num_class': num_class,
             'learning_rate': 0.2, 'n_jobs': -1, 'min_child_weight': 10}
    before = datetime.datetime.now()
    result = xgb.cv(param, dtrain, num_boost_round=boost_rounds, nfold=cv_fold, metrics=['merror'], stratified=True,
                    shuffle=True)
    after = datetime.datetime.now()
    print("XGBoost")
    xgb_print(result, before, after)


def xgboost_rf(data, label):
    # parametre: https://xgboost.readthedocs.io/en/latest/parameter.html
    dtrain = xgb.DMatrix(data, label=label)
    param = {'max_depth': 7, 'objective': 'multi:softmax', 'eval_metric': 'merror', 'num_class': num_class,
             'learning_rate': 0.2, 'n_jobs': -1, 'num_parallel_tree': 100}
    before = datetime.datetime.now()
    result = xgb.cv(param, dtrain, num_boost_round=1, nfold=cv_fold, metrics=['merror'], stratified=True, shuffle=True)
    after = datetime.datetime.now()
    print("XGBoost forest")
    xgb_print(result, before, after)


def hist_xgboost(data, label):
    dtrain = xgb.DMatrix(data, label=label)
    param = {'max_depth': 7, 'objective': 'multi:softmax', 'eval_metric': 'merror', 'num_class': num_class,
             'learning_rate': 0.2, 'n_jobs': -1, 'tree_method': 'hist'}
    before = datetime.datetime.now()
    result = xgb.cv(param, dtrain, num_boost_round=boost_rounds, nfold=cv_fold, metrics=['merror'], stratified=True,
                    shuffle=True)
    after = datetime.datetime.now()
    print("XGBoost hist")
    xgb_print(result, before, after)


def linear_xgb(data, label):  # implicitna selekcia atributov
    dtrain = xgb.DMatrix(data, label=label)
    param = {'max_depth': 7, 'objective': 'multi:softmax', 'eval_metric': 'merror', 'num_class': num_class,
             'learning_rate': 0.2, 'n_jobs': -1, 'booster': 'gblinear',
             'feature_selector': 'thrifty', 'updater': 'coord_descent'}
    before = datetime.datetime.now()
    cvresult = xgb.cv(param, dtrain, num_boost_round=boost_rounds, nfold=cv_fold, metrics=['merror'], stratified=True,
                      shuffle=True)
    after = datetime.datetime.now()
    print("linear XGBoost")
    xgb_print(cvresult, before, after)


def xgboost_dart(data, label):
    dtrain = xgb.DMatrix(data, label=label)
    param = {'max_depth': 7, 'objective': 'multi:softmax', 'eval_metric': 'merror', 'num_class': num_class,
             'learning_rate': 0.2, 'n_jobs': -1, 'min_child_weight': 10, 'booster': 'dart',
             'rate_drop': 0.1, 'skip_drop': 0.5}
    before = datetime.datetime.now()
    result = xgb.cv(param, dtrain, num_boost_round=boost_rounds, nfold=cv_fold, metrics=['merror'], stratified=True,
                    shuffle=True)
    after = datetime.datetime.now()
    print("XGBoost dart")
    xgb_print(result, before, after)


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
             "num_class": num_class, "metric": 'multi_error', 'min_data_in_leaf': 10, 'num_threads': -1,
             'verbosity': -1,
             'min_data_in_bin': 3, 'max_bin': 255, 'enable_bundle': True, 'max_conflict_rate': 0.0}
    before = datetime.datetime.now()
    result = lgb.cv(param, dtrain, num_boost_round=boost_rounds, nfold=cv_fold, stratified=True, verbose_eval=None,
                    shuffle=True)
    after = datetime.datetime.now()
    print("LGBM")
    LGBM_print(result, before, after)


def LGBM_rf(data, label):
    # https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
    dtrain = lgb.Dataset(data=data, label=label)
    param = {"max_depth": 7, "learning_rate": 0.2, "objective": 'multiclass', 'num_leaves': 80, 'boosting': 'rf',
             "num_class": num_class, "metric": 'multi_error', 'min_data_in_leaf': 10, 'num_threads': -1,
             'verbosity': -1,
             'bagging_fraction': 0.9, 'bagging_freq': 10, 'num_trees': 100}
    before = datetime.datetime.now()
    result = lgb.cv(param, dtrain, nfold=cv_fold, stratified=True, verbose_eval=None, shuffle=True)
    after = datetime.datetime.now()
    print("LGBM forest")
    LGBM_print(result, before, after)


def LGBM_goss(data, label):
    # https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
    dtrain = lgb.Dataset(data=data, label=label)
    param = {"max_depth": 7, "learning_rate": 0.2, "objective": 'multiclass', 'num_leaves': 80, 'boosting': 'goss',
             "num_class": num_class, "metric": 'multi_error', 'min_data_in_leaf': 10, 'num_threads': -1,
             'verbosity': -1,
             'min_data_in_bin': 3, 'max_bin': 255, 'enable_bundle': True, 'max_conflict_rate': 0.0}
    before = datetime.datetime.now()
    result = lgb.cv(param, dtrain, num_boost_round=boost_rounds, nfold=cv_fold, stratified=True, verbose_eval=None,
                    shuffle=True)
    after = datetime.datetime.now()
    print("LGBM GOSS")
    LGBM_print(result, before, after)


def LGBM_dart(data, label):
    # https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
    dtrain = lgb.Dataset(data=data, label=label)
    param = {"max_depth": 7, "learning_rate": 0.2, "objective": 'multiclass', 'num_leaves': 80, 'boosting': 'dart',
             "num_class": num_class, "metric": 'multi_error', 'min_data_in_leaf': 10, 'num_threads': -1,
             'verbosity': -1,
             'min_data_in_bin': 3, 'max_bin': 255, 'xgboost_dart_mode': False}
    before = datetime.datetime.now()
    result = lgb.cv(param, dtrain, num_boost_round=boost_rounds, nfold=cv_fold, stratified=True, verbose_eval=None,
                    shuffle=True)
    after = datetime.datetime.now()
    print("LGBM dart")
    LGBM_print(result, before, after)


def LGBM_print(result, before, after):
    print("najlepsi priemer: " + str(1 - min(result['multi_error-mean'])))
    print("index najlepsieho: " + str(result['multi_error-mean'].index(min(result['multi_error-mean']))))
    print("najhorsi priemer: " + str(1 - max(result['multi_error-mean'])))
    print("finalny priemer: " + str(1 - result['multi_error-mean'][-1]))
    print("cas: " + str(after - before))
    print('\n')


def CAT(data, label):
    pool = cat.Pool(data, label, has_header=False)
    params = {
        "loss_function": 'MultiClassOneVsAll', "eval_metric": 'MultiClassOneVsAll', "max_depth": 7,
        "learning_rate": 0.2, "classes_count": num_class, "task_type": 'CPU', "thread_count": 6, "verbose_eval": False}
    before = datetime.datetime.now()
    results = cat.cv(pool=pool, params=params, num_boost_round=boost_rounds, fold_count=cv_fold, shuffle=True,
                     stratified=True,
                     verbose=False)
    after = datetime.datetime.now()
    print("CatBoost")
    print("najlepsi priemer: " + str(1 - min(results['test-MultiClassOneVsAll-mean'])))
    print("index najlepsieho: " + str(results['test-MultiClassOneVsAll-mean'][results['test-MultiClassOneVsAll-mean']
                                                                              == min(
        results['test-MultiClassOneVsAll-mean'])].index[0]))
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
    # print("najlepsi vysledok: " + str(max(scores)))
    # print("najhorsi vysledok: " + str(min(scores)))
    print("priemerny vysledok: " + str(statistics.mean(scores)))
    print("cas: " + str(after - before))
    print('\n')


def RGF(data, label, method):
    rgf = RGFClassifier(max_leaf=1000, algorithm=method, verbose=False, n_jobs=-1, min_samples_leaf=10,
                        learning_rate=0.2)  # max_leaf je globalne pre cely les co je defaultne 50 stromov
    # multi_cv_fit(rgf, multi_cv_repeats, cv_fold, data, label)
    print(method)
    cv_fit(rgf, cv_fold, data, label)


def RFC(data, label):
    rfc = RandomForestClassifier(n_estimators=1000, max_depth=7, min_samples_leaf=10, max_leaf_nodes=100,
                                 bootstrap=True, n_jobs=-1, warm_start=False)
    print("RFC")
    # multi_cv_fit(rfc, multi_cv_repeats, cv_fold, data, label)
    cv_fit(rfc, cv_fold, data, label)


def SGD(data, label):
    sgd = SGDClassifier(
        loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, max_iter=max_iter, tol=0.001, shuffle=False, verbose=0,
        n_jobs=-1, learning_rate='optimal', eta0=0.0, power_t=0.5)
    print("SGD SVC")
    # multi_cv_fit(sgd, multi_cv_repeats, cv_fold, data, label)
    cv_fit(sgd, cv_fold, data, label)


def LSVC(data, label):
    svc = LinearSVC(penalty='l1', loss='squared_hinge', dual=False, tol=0.001, C=1, multi_class='ovr',
                    fit_intercept=False, verbose=0, max_iter=max_iter)
    # ak mam standardizovane data tak fit_intercept mozem dat na False
    # ak mam vela atributov tak dam dual na True
    print("linear SVC [liblinear]")
    # multi_cv_fit(svc, multi_cv_repeats, cv_fold, data, label)
    cv_fit(svc, cv_fold, data, label)


def SVM(data, label, kernel, message):
    svm = SVC(C=1.0, kernel=kernel, shrinking=True, probability=False, tol=0.001, cache_size=200, verbose=False,
              max_iter=max_iter, decision_function_shape='ovr', gamma='scale')
    print(message)
    # multi_cv_fit(svm, multi_cv_repeats, cv_fold, data, label)
    cv_fit(svm, cv_fold, data, label)


def create_original_dataset_for_cluster_dataset():
    #  pouzival som na odstranenie outlierov z konsenzoveho datastu aby som vedel porovnat obe metody labelingu
    #  na rovnakych datach. Po porovani to uz nepouivam
    cluster_labels = np.loadtxt(cluster_labels_outliers_path, delimiter=',', skiprows=1, dtype=np.int8)
    to_delete = []
    for i in range(len(cluster_labels)):
        if cluster_labels[i] == -1:
            to_delete.append(i)
    labels = np.loadtxt(labels_path, delimiter=',', skiprows=1, dtype=np.uint8)
    labels = np.delete(labels, to_delete)
    data = np.loadtxt(feature_file, delimiter=',', skiprows=1, dtype=np.uint64)
    data = np.delete(data, to_delete, axis=0)
    standard_data = np.loadtxt(standard_feature_file, delimiter=',', skiprows=1, dtype=np.float64)
    standard_data = np.delete(standard_data, to_delete, axis=0)
    return labels, data, standard_data


# tieto metody pouzijem az po tom, co urobim klasifikaciu selektovaneho datasetu a nastavim hyperparametre
def xgboost_train(data, labels, selector):
    dtrain = xgb.DMatrix(data, labels)
    param = {'max_depth': 7, 'objective': 'multi:softmax', 'eval_metric': 'merror', 'num_class': num_class,
             'learning_rate': 0.2, 'n_jobs': -1, 'min_child_weight': 10}
    result = xgb.train(param, dtrain, num_boost_round=boost_rounds)
    result.save_model(trained_path + 'xgb' + '_from_' + selector + '.txt')


def lgbm_train(data, labels, selector):
    param = {"max_depth": 7, "learning_rate": 0.2, "objective": 'multiclass', 'num_leaves': 80,
             "num_class": num_class, "metric": 'multi_error', 'min_data_in_leaf': 10, 'num_threads': -1,
             'verbosity': -1,
             'min_data_in_bin': 3, 'max_bin': 255, 'enable_bundle': True, 'max_conflict_rate': 0.0}
    dtrain = lgb.Dataset(data, labels)
    result = lgb.train(param, dtrain, num_boost_round=boost_rounds, verbose_eval=None)  # cv nevracia model,len vysledky
    result.save_model(trained_path + 'lgb' + '_from_' + selector + '.txt', num_iteration=result.best_iteration)


def svc_train(data, labels, selector):
    svc = LinearSVC(penalty='l1', loss='squared_hinge', dual=False, tol=0.001, C=1, multi_class='ovr',
                    fit_intercept=False, verbose=0, max_iter=max_iter)
    svc.fit(data, labels)
    with open(trained_path + 'svc' + '_from_' + selector + '.txt', 'wb') as subor:
        pickle.dump(svc, subor, protocol=pickle.HIGHEST_PROTOCOL)


def train_for_prediction():  # trenovat musim na selekciach a potom pouzit pri predikcii spravny model
    labels = np.loadtxt(labels_path, delimiter=',', skiprows=1, dtype=np.uint8)
    files = glob.glob(selected_dir)
    for file in files:
        data = np.loadtxt(file, delimiter=',', skiprows=1, dtype=np.uint64)
        lgbm_train(data, labels, os.path.basename(file)[:-4])
        xgboost_train(data, labels, os.path.basename(file)[:-4])
    files = glob.glob(standard_selected_dir + '*')
    for file in files:
        standard_data = np.loadtxt(file, delimiter=',', skiprows=1, dtype=np.float64)
        svc_train(standard_data, labels, os.path.basename(file)[:-4])


def xgboost_predict(data, selector):
    data = xgb.DMatrix(data)
    model = xgb.Booster()
    model.load_model(trained_path + 'xgb' + '_from_' + selector + '.txt')
    result = model.predict(data)
    return result


def lgbm_predict(data, selector):
    model = lgb.Booster(model_file=trained_path + 'lgb' + '_from_' + selector + '.txt')
    probs = model.predict(data)  # vracia pravdepodobnosti pre kazdu triedu
    result = []
    for i in range(len(probs)):
        max_prob_index = (np.where(probs == np.max(probs[i]))[1])[0]
        result.append(max_prob_index)
    return result


def svc_predict(data, selector):
    with open(trained_path + 'svc' + '_from_' + selector + '.txt', 'rb') as subor:
        model = pickle.load(subor)
    result = model.predict(data)
    return result


def predictions():  # na predikovanom datasete - vsetky selekcie
    with open(predictions_file, "w", newline='') as output:
        writer = csv.writer(output, delimiter=',')
        files = glob.glob(selected_dir)
        for file in files:
            data = np.loadtxt(file, delimiter=',', skiprows=1, dtype=np.uint64)
            sel = VarianceThreshold()
            try:
                sel.fit_transform(data)
                support = sel.get_support(True)
                print("number of irrelevant features for " + os.path.basename(file)[:-4] + " is " +
                      str(len(data[0]) - len(support)))
            except ValueError:
                print("empty")
            selector = os.path.basename(file)[:-4]
            writer.writerow([selector])
            result = lgbm_predict(data, selector)
            writer.writerow(result)
            result = xgboost_predict(data, selector)
            result = result.astype(np.uint8)
            writer.writerow(result)
            data = np.loadtxt(standard_selected_dir + os.path.basename(file), delimiter=',', skiprows=1,
                              dtype=np.float64)
            print("standard")
            sel = VarianceThreshold()
            try:
                sel.fit_transform(data)
                support = sel.get_support(True)
                print("number of irrelevant features for " + os.path.basename(file)[:-4] + " is " +
                      str(len(data[0]) - len(support)))
            except ValueError:
                print("empty")
            result = svc_predict(data, selector)
            writer.writerow(result)
        #     writer.writerow("-----------------")
        # files = glob.glob(standard_selected_dir + '*')
        # for file in files:
        #     standard_data = np.loadtxt(file, delimiter=',', skiprows=1, dtype=np.float64)
        #     writer.writerow(os.path.basename(file)[:-4])
        #     result = svc_predict(standard_data)
        #     writer.writerow(result)


def tree_methods(data, labels):
    xgboost(data, labels)
    # LGBM_goss(data, labels)
    LGBM(data, labels)
    # RGF(data, labels, "RGF")
    # RGF(data, labels, "RGF_Opt")
    # RFC(data, labels)  # fast


def svm_methods(data, labels):
    # SVM(data, labels, 'rbf', "RBF SVC")
    # SVM(data, labels, 'linear', "linear SVC [libsvm]")  # fast
    # SVM(data, labels, 'sigmoid', "sigmoid SVC")
    # SGD(data, labels)  # fast
    LSVC(data, labels)  # fast


def run_methods():
    sys.stdout = open(times_path + 'classification_times.txt', 'w')
    labels = np.loadtxt(labels_path, delimiter=',', skiprows=1, dtype=np.uint8)
    # labels = create_original_labels_for_cluster_dataset()  # pre porovnanie povodnych labels na cluster dataset
    data = np.loadtxt(feature_file, delimiter=',', skiprows=1, dtype=np.uint64)
    print("vsetky data: " + str(len(data[0])))
    print('\n')
    tree_methods(data, labels)
    standard_data = np.loadtxt(standard_feature_file, delimiter=',', skiprows=1, dtype=np.float64)
    svm_methods(standard_data, labels)
    sys.stdout.close()


def check_selections():
    sys.stdout = open(classification_selected_dir + 'classification_selected.txt', 'w')
    files = glob.glob(selected_dir)
    labels = np.loadtxt(labels_path, delimiter=',', skiprows=1, dtype=np.uint8)
    for file in files:
        data = np.loadtxt(file, delimiter=',', skiprows=1, dtype=np.uint64)
        print(os.path.basename(file)[:-4] + ": " + str(len(data[0])))
        tree_methods(data, labels)
        print("------------------------------------------------------------")
        print('\n')
    preprocessing.standardize(selected_dir, standard_selected_dir, True)  # aj ulozi scaler pre kazdu selekciu
    files = glob.glob(standard_selected_dir + '*')
    for file in files:
        standard_data = np.loadtxt(file, delimiter=',', skiprows=1, dtype=np.float64)
        print(os.path.basename(file)[:-4] + ": " + str(len(standard_data[0])))
        svm_methods(standard_data, labels)
        print("------------------------------------------------------------")
        print('\n')
    sys.stdout.close()


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if mode == "preselection":
            run_methods()  # pred selekciou
        if mode == "selection_check":
            check_selections()  # po selekcii
        if mode == "prediction_train":
            train_for_prediction()  # pred predikciou, na selekciach
        if mode == "prediction":
            predictions()  # predikcia, na novom datasete
    sys.stdout.close()


if __name__ == "__main__":
    main()
