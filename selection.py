from skfeature.function.statistical_based import CFS, gini_index
from skfeature.function.information_theoretical_based import FCBF, MIFS, MRMR, CIFE, JMI, CMIM, DISR
from skfeature.function.similarity_based import SPEC, fisher_score, reliefF, trace_ratio, lap_score
import numpy as np
import csv
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif, SelectFromModel
import pandas as pd
from functools import partial
import lightgbm as lgb
import datetime
import xgboost as xgb
from rgf.sklearn import RGFClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
import catboost as cat
from pyHSICLasso import HSICLasso
import sys
import glob
import os

feature_path = 'features/very_simple.csv'
standard_feature_path = 'features/standard/very_simple.csv'
labels_path = 'subory/clear_labels2_head.csv'
output_dir = 'features/selection/'

sys.stdout = open('vysledky/selection_times.txt', 'w')
np.set_printoptions(threshold=np.inf)


def numpy_load():
    labels = np.loadtxt(labels_path, delimiter=',', skiprows=1, dtype=np.uint8)
    data = np.loadtxt(feature_path, delimiter=',', skiprows=1, dtype=np.uint64)
    header = np.loadtxt(feature_path, delimiter=',', max_rows=1, dtype="str")
    return data, header, labels


def pandas_load():
    labels = pd.read_csv(labels_path, dtype=np.int8)  # pri sklearn treba mat label.values.ravel()
    # standard_data = pandas.read_csv(standard_feature_path, dtype=np.float32)
    data = pd.read_csv(feature_path, skiprows=1, header=None)
    # ak mam header tak pri niektorych atributoch ma xgboost problemy lebo obsahuju nepovolene znaky, preto mam none
    header = pd.read_csv(feature_path, nrows=1, header=None)
    header = header.to_numpy()[0]
    return data, header, labels.values.ravel()


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


def cfs():  # extremne pomaly
    # http://featureselection.asu.edu/html/skfeature.function.statistical_based.CFS.html
    before = datetime.datetime.now()
    result = CFS.cfs(data, labels, mode="index", n_selected_features=treshold)
    after = datetime.datetime.now()
    print("CFS")
    print(len(result))
    print("cas: " + str(after - before))
    print('\n')
    if len(result) < len(header):
        transform_and_save(result, "CFS")


def mrmr():
    before = datetime.datetime.now()
    result = MRMR.mrmr(data, labels, mode="index", n_selected_features=treshold)
    after = datetime.datetime.now()
    print("mRMR")
    print(len(result))
    print("cas: " + str(after - before))
    print('\n')
    if len(result) < len(header):
        transform_and_save(result, "MRMR")


def cife():
    before = datetime.datetime.now()
    result = CIFE.cife(data, labels, mode="index", n_selected_features=treshold)
    after = datetime.datetime.now()
    print("CIFE")
    print(len(result))
    print("cas: " + str(after - before))
    print('\n')
    if len(result) < len(header):
        transform_and_save(result, "CIFE")


def jmi():
    before = datetime.datetime.now()
    result = JMI.jmi(data, labels, mode="index", n_selected_features=treshold)
    # algoritmus ma treshold ale pri testoch ho nedosiahol a vyberal vsetky hodnoty
    after = datetime.datetime.now()
    print("JMI")
    print(len(result))
    print("cas: " + str(after - before))
    print('\n')
    if len(result) < len(header):
        transform_and_save(result, "JMI")


def cmim():
    before = datetime.datetime.now()
    result = CMIM.cmim(data, labels, mode="index", n_selected_features=treshold)
    after = datetime.datetime.now()
    print("CMIM")
    print(len(result))
    print("cas: " + str(after - before))
    print('\n')
    if len(result) < len(header):
        transform_and_save(result, "CMIM")


def disr():
    before = datetime.datetime.now()
    result = DISR.disr(data, labels, mode="index", n_selected_features=treshold)
    # algoritmus ma treshold ale pri testoch ho nedosiahol a vyberal vsetky hodnoty
    after = datetime.datetime.now()
    print("DISR")
    print(len(result))
    print("cas: " + str(after - before))
    print('\n')
    if len(result) < len(header):
        transform_and_save(result, "DISR")


# rychle --------------------
def fcbf():
    # http://featureselection.asu.edu/html/skfeature.function.information_theoretical_based.FCBF.html
    before = datetime.datetime.now()
    result = FCBF.fcbf(data, labels, mode="index", delta=0)  # treshold je delta
    after = datetime.datetime.now()
    print("FCBF")
    print(len(result))
    print("cas: " + str(after - before))
    print('\n')
    if len(result) < len(header):
        transform_and_save(result, "FCBF")


def mifs():
    before = datetime.datetime.now()
    result = MIFS.mifs(data, labels, mode="index", n_selected_features=treshold)
    after = datetime.datetime.now()
    print("MIFS")
    print(len(result))
    print("cas: " + str(after - before))
    print('\n')
    if len(result) < len(header):
        transform_and_save(result, "MIFS")


# na prvotnu selekciu (velmi rychle, ale nezarucuju vyhodenie redundantnych aj irelevantnych)
# -------------------------------------
def fit(selector, prefix):
    before = datetime.datetime.now()
    selector.fit(data, labels)
    after = datetime.datetime.now()
    new_data = selector.transform(data)
    selected = selector.get_support(True)
    print(len(selected))
    print("cas: " + str(after - before))
    print('\n')
    if len(selected) < len(header):
        save_to_csv(new_data, selected, prefix)


def chi_square(treshold):
    sel = SelectKBest(chi2, k=treshold)
    print("Chi-square")
    fit(sel, "chi-square")


def MI(treshold):
    sel = SelectKBest(score_func=partial(mutual_info_classif, discrete_features=True), k=treshold)
    print("Mutual information")
    fit(sel, "mutual_info")


def f_anova(treshold):
    sel = SelectKBest(f_classif, k=treshold)
    print("ANOVA F-score")
    fit(sel, "ANOVA_F-score")


def gini(treshold):
    before = datetime.datetime.now()
    result = gini_index.gini_index(data, labels, mode="index")
    after = datetime.datetime.now()
    print("Gini")
    result = result[:treshold]
    print(len(result))
    print("cas: " + str(after - before))
    print('\n')
    if len(result) < len(header):
        transform_and_save(result, "Gini")


def fisher(treshold):
    before = datetime.datetime.now()
    result = fisher_score.fisher_score(data, labels, mode="index")
    after = datetime.datetime.now()
    print("Fisher")
    result = result[:treshold]
    print(len(result))
    print("cas: " + str(after - before))
    print('\n')
    if len(result) < len(header):
        transform_and_save(result, "Fisher")


def lap(treshold):
    before = datetime.datetime.now()
    result = lap_score.lap_score(data.copy(), labels.copy(), mode="index")  # prepisuje vstup, preto ho kopirujem
    after = datetime.datetime.now()
    print("Laplacian")
    result = result[:treshold]
    print(len(result))
    print("cas: " + str(after - before))
    print('\n')
    if len(result) < len(header):
        transform_and_save(result, "Laplacian")


def trace(treshold):
    before = datetime.datetime.now()
    result = trace_ratio.trace_ratio(data, labels, mode="index", n_selected_features=treshold)
    after = datetime.datetime.now()
    print("Trace ratio")
    # result = result[:treshold]
    print(len(result))
    print("cas: " + str(after - before))
    print('\n')
    if len(result) < len(header):
        transform_and_save(result, "Trace_ratio")


def spec(treshold):
    before = datetime.datetime.now()
    result = SPEC.spec(data, mode="index")
    after = datetime.datetime.now()
    print("SPEC")
    result = result[:treshold]
    print(len(result))
    print("cas: " + str(after - before))
    print('\n')
    if len(result) < len(header):
        transform_and_save(result, "SPEC")


def relieff(treshold):
    before = datetime.datetime.now()
    result = reliefF.reliefF(data, labels, mode="index")
    after = datetime.datetime.now()
    print("relieff")
    result = result[:treshold]
    print(len(result))
    print("cas: " + str(after - before))
    print('\n')
    if len(result) < len(header):
        transform_and_save(result, "ReliefF")
# -------------------------------------


def model_selection_treshold(model, prefix):
    selection = SelectFromModel(model, threshold=-np.inf, prefit=True, max_features=treshold)
    new_data = selection.transform(data)
    selected = selection.get_support(True)
    print(len(selected))
    if len(selected) < len(header):
        save_to_csv(new_data, selected, prefix)


def model_selection_zero(model, prefix):
    selection = SelectFromModel(model, threshold=1e-5, prefit=True)
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
    # model_selection_median(model, prefix)
    # model_selection_zero(model, prefix)  # len pre L1 a elasticnet SVM
    print("cas: " + str(after - before))
    print('\n')


def xgboost():
    # ak budu prve tri pomale tak dam n_estimators na 50
    model_gain = xgb.XGBClassifier(max_depth=7, objective='multi:softmax', min_child_weight=10, learning_rate=0.2,
                                   n_jobs=-1, n_estimators=100, importance_type='gain', num_class=10)  # 'total_gain'
    model_cover = xgb.XGBClassifier(max_depth=7, objective='multi:softmax', min_child_weight=10, learning_rate=0.2,
                                    n_jobs=-1, n_estimators=100, importance_type='weight', num_class=10)
    print("XGBoost gain")
    model_fit(model_gain, "XGBoost gain")
    print("XGBoost split")
    model_fit(model_cover, "XGBoost split")


def LGBM():
    model_split = lgb.LGBMClassifier(max_depth=7, learning_rate=0.2, n_estimators=100, objective='multiclass',
                                     n_jobs=-1, num_leaves=80, min_child_samples=10, importance_type='split',  num_class=10)
    model_gain = lgb.LGBMClassifier(max_depth=7, learning_rate=0.2, n_estimators=100, objective='multiclass',
                                    n_jobs=-1, num_leaves=80, min_child_samples=10, importance_type='gain',  num_class=10)
    print("LGBM split")
    model_fit(model_split, "LGBM split")
    print("LGBM gain")
    model_fit(model_gain, "LGBM gain")


def CAT():
    model = cat.CatBoostClassifier(max_depth=7, n_estimators=100, loss_function='MultiClassOneVsAll', learning_rate=0.2,
                                   task_type='CPU', verbose=False, thread_count=6, classes_count=10)
    print("CatBoost")
    model_fit(model, "CatBoost")


def RGF():
    model = RGFClassifier(max_leaf=1000, algorithm="RGF_Opt", verbose=False, n_jobs=-1, min_samples_leaf=10,
                          learning_rate=0.2)
    print("RGF")
    model_fit(model, "RGF")


def RFC():
    model = RandomForestClassifier(n_estimators=1000, max_depth=7, min_samples_leaf=10, max_leaf_nodes=100,
                                   bootstrap=True, n_jobs=-1, warm_start=False)
    print("RFC")
    model_fit(model, "RFC")


def LSVC_l1():
    model_l1 = LinearSVC(penalty='l1', loss='squared_hinge', dual=False, tol=0.001, C=1, multi_class='ovr',
                         fit_intercept=False, verbose=0, max_iter=1000)
    # ak mam standardizovane data tak fit_intercept mozem dat na False
    # ak mam vela atributov tak dam dual na True
    print("SVC L1")
    model_fit(model_l1, "SVC_L1")


def LSVC_l2():
    model_l2 = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.001, C=1, multi_class='ovr',
                         fit_intercept=False, verbose=0, max_iter=1000)
    print("SVC L2")
    model_fit(model_l2, "SVC_L2")


def SGD_l1():
    model_l1 = SGDClassifier(loss='hinge', penalty='l1', alpha=0.0001, max_iter=1000, tol=0.001,
                             shuffle=True, verbose=0, n_jobs=-1, learning_rate='optimal', eta0=0.0, power_t=0.5)
    print("SGD L1")
    model_fit(model_l1, "SGD_L1")


def SGD_l2():
    model_l2 = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, max_iter=1000, tol=0.001,
                             shuffle=True, verbose=0, n_jobs=-1, learning_rate='optimal', eta0=0.0, power_t=0.5)
    print("SGD L2")
    model_fit(model_l2, "SGD_L2")


def SGD_elastic():
    model_elastic = SGDClassifier(loss='hinge', penalty='elasticnet', alpha=0.0001, l1_ratio=0.15, max_iter=1000,
                                  tol=0.001, shuffle=True, verbose=0, n_jobs=-1, learning_rate='optimal', eta0=0.0,
                                  power_t=0.5)
    print("SGD elasticnet")
    model_fit(model_elastic, "SGD_elasticnet")


def SVM():
    svm = SVC(C=1.0, kernel="linear", shrinking=True, probability=False, tol=0.001, cache_size=200, verbose=False,
              max_iter=1000, decision_function_shape='ovr', gamma='scale')
    print("SVM linear")
    model_fit(svm, "SVM")


def HSIC_lasso(treshold):
    hsic = HSICLasso()
    hsic.input(data, labels)
    before = datetime.datetime.now()
    hsic.classification(num_feat=treshold, B=0, M=1, max_neighbors=10, discrete_x=False)
    # B a M su na postupne nacitanie ak mam velky dataset, B deli pocet vzoriek, pre klasicky algoritmus B=0, M=1
    after = datetime.datetime.now()
    print("HSIC Lasso")
    selected = hsic.get_index()
    print(len(selected))
    print("cas: " + str(after - before))
    print('\n')
    if len(selected) < len(header):
        transform_and_save(selected, "HSIC_Lasso")


def select_best_n(n, input_path, output_path):
    files = sorted(glob.glob(input_path + "*"))
    for name in files:
        with open(name, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)
            data = list(reader)
        with open(output_path + os.path.basename(f.name), "w", newline='') as csv_file:  # zapisem len vybrane
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(header[:n])
            for row in data:
                writer.writerow(row[:n])


def for_small_data():
    mifs()
    mrmr()
    cife()
    jmi()
    cmim()
    disr()
    trace(treshold)
    CAT()
    RGF()


def for_big_data():
    chi_square(treshold)
    MI(treshold)
    f_anova(treshold)
    gini(treshold)
    fisher(treshold)
    lap(treshold)
    spec(treshold)
    relieff(treshold)
    xgboost()
    LGBM()
    RFC()


def svm_big_data():
    LSVC_l1()
    SGD_l1()
    SGD_l2()
    SGD_elastic()
    SVM()


labels = np.loadtxt(labels_path, delimiter=',', skiprows=1, dtype=np.uint8)
header = np.loadtxt(feature_path, delimiter=',', max_rows=1, dtype="str")
data = np.loadtxt(feature_path, delimiter=',', skiprows=1, dtype=np.uint64)
print("pocet atributov: " + str(len(header)))
print('\n')
treshold = 85

# cfs()
# fcbf()
HSIC_lasso(treshold)
#
# for_small_data()
# for_big_data()
#
# output_dir = 'features/selection_standard/'
# header = np.loadtxt(standard_feature_path, delimiter=',', max_rows=1, dtype="str")
# data = np.loadtxt(standard_feature_path, delimiter=',', skiprows=1, dtype=np.float64)
#
# svm_big_data()
# LSVC_l2()

# select_best_n(79, output_dir, "features/best_n/")
sys.stdout.close()
