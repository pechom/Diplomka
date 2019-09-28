from skfeature.function.statistical_based import CFS, gini_index
from skfeature.function.information_theoretical_based import FCBF, MIFS, MRMR, CIFE, JMI, CMIM, DISR
from skfeature.function.similarity_based import SPEC, fisher_score, reliefF, trace_ratio, lap_score
from sklearn.feature_selection import SelectPercentile, chi2, mutual_info_classif, f_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
import lightgbm as lgb
import xgboost as xgb
from rgf.sklearn import RGFClassifier
import catboost as cat
from pyHSICLasso import HSICLasso
import numpy as np
import csv
import pandas as pd
from functools import partial
import datetime
import sys
import glob
import os

feature_path = 'seminar/features.csv'
standard_feature_path = 'seminar/simple_standard.csv'
labels_path = 'seminar/clear_labels2_head.csv'
output_dir = "seminar/selection/"
intersections_path = 'seminar/intersections.txt'
sys.stdout = open('seminar/selection_times.txt', 'w')
np.set_printoptions(threshold=np.inf)


def numpy_load():
    labels = np.loadtxt(labels_path, delimiter=',', skiprows=1, dtype=np.int8)
    data = np.loadtxt(feature_path, delimiter=',', skiprows=1)
    header = pd.read_csv(feature_path, nrows=1, header=None)
    header = header.to_numpy()[0]
    return data, header, labels


def save_to_csv(transformed_data, selected, prefix, path=output_dir):
    with open(path + prefix + ".csv", "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(header[selected])
        writer.writerows(transformed_data)


def transform_and_save(selected, prefix, path=output_dir):
    with open(path + prefix + ".csv", "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(header[selected])
        writer.writerows(data[:, selected])


def cfs():
    # http://featureselection.asu.edu/html/skfeature.function.statistical_based.CFS.html
    before = datetime.datetime.now()
    result = CFS.cfs(data, labels, mode="index")
    after = datetime.datetime.now()
    print("CFS")
    print(len(result))
    print("cas: " + str(after - before))
    print('\n')
    if len(result) < len(header):
        transform_and_save(result, "CFS")


def mrmr():
    before = datetime.datetime.now()
    result = MRMR.mrmr(data, labels, mode="index")
    after = datetime.datetime.now()
    print("mRMR")
    print(len(result))
    print("cas: " + str(after - before))
    print('\n')
    if len(result) < len(header):
        transform_and_save(result, "MRMR")


def cife():
    before = datetime.datetime.now()
    result = CIFE.cife(data, labels, mode="index")
    after = datetime.datetime.now()
    print("CIFE")
    print(len(result))
    print("cas: " + str(after - before))
    print('\n')
    if len(result) < len(header):
        transform_and_save(result, "CIFE")


def jmi():
    before = datetime.datetime.now()
    result = JMI.jmi(data, labels, mode="index")
    after = datetime.datetime.now()
    print("JMI")
    print(len(result))
    print("cas: " + str(after - before))
    print('\n')
    if len(result) < len(header):
        transform_and_save(result, "JMI")


def cmim():
    before = datetime.datetime.now()
    result = CMIM.cmim(data, labels, mode="index")
    after = datetime.datetime.now()
    print("CMIM")
    print(len(result))
    print("cas: " + str(after - before))
    print('\n')
    if len(result) < len(header):
        transform_and_save(result, "CMIM")


def disr():
    before = datetime.datetime.now()
    result = DISR.disr(data, labels, mode="index")
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
    result = FCBF.fcbf(data, labels, mode="index")
    after = datetime.datetime.now()
    print("FCBF")
    print(len(result))
    print("cas: " + str(after - before))
    print('\n')
    if len(result) < len(header):
        transform_and_save(result, "FCBF")


def mifs():
    before = datetime.datetime.now()
    result = MIFS.mifs(data, labels, mode="index")
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


def chi_square(percentile):
    sel = SelectPercentile(chi2, percentile=percentile)
    print("Chi-square")
    fit(sel, "chi-square")


def MI(percentile):
    sel = SelectPercentile(score_func=partial(mutual_info_classif, discrete_features=True), percentile=percentile)
    print("Mutual information")
    fit(sel, "mutual_info")


def f_anova(percentile):
    sel = SelectPercentile(f_classif, percentile=percentile)
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
    result = lap_score.lap_score(data.copy(), mode="index")  # prepisuje vstup
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
    result = trace_ratio.trace_ratio(data, labels, mode="index")
    after = datetime.datetime.now()
    print("Trace ratio")
    result = result[:treshold]
    print(len(result))
    print("cas: " + str(after - before))
    print('\n')
    if len(result) < len(header):
        transform_and_save(result, "Trace_ratio")


# prilis pomale, viac ako sekunda na 188
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


def model_selection_mean(model, prefix):
    selection = SelectFromModel(model, threshold='mean', prefit=True)
    new_data = selection.transform(data)
    selected = selection.get_support(True)
    print(len(selected))
    if len(selected) < len(header):
        save_to_csv(new_data, selected, prefix)


def model_selection_median(model, prefix):
    selection = SelectFromModel(model, threshold='median', prefit=True)
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
    model_selection_mean(model, prefix)
    # model_selection_median(model, prefix)
    # model_selection_zero(model, prefix)  # len pre L1 a elasticnet SVM
    print("cas: " + str(after - before))
    print('\n')


def xgboost():
    model_gain = xgb.XGBClassifier(max_depth=7, objective='multi:softmax', min_child_weight=10, learning_rate=0.2,
                                   n_jobs=-1, n_estimators=100, importance_type='gain')
    model_cover = xgb.XGBClassifier(max_depth=7, objective='multi:softmax', min_child_weight=10, learning_rate=0.2,
                                    n_jobs=-1, n_estimators=100, importance_type='total_cover')
    print("XGBoost gain")
    model_fit(model_gain, "XGBoost gain")
    print("XGBoost total cover")
    model_fit(model_cover, "XGBoost total cover")


def LGBM():
    model_split = lgb.LGBMClassifier(max_depth=7, learning_rate=0.2, n_estimators=100, objective='multiclass',
                                     n_jobs=-1, num_leaves=80, min_child_samples=10, importance_type='split')
    model_gain = lgb.LGBMClassifier(max_depth=7, learning_rate=0.2, n_estimators=100, objective='multiclass',
                                    n_jobs=-1, num_leaves=80, min_child_samples=10, importance_type='gain')
    print("LGBM split")
    model_fit(model_split, "LGBM split")
    print("LGBM gain")
    model_fit(model_gain, "LGBM gain")


def CAT():
    model = cat.CatBoostClassifier(max_depth=7, n_estimators=100, loss_function='MultiClassOneVsAll', learning_rate=0.2,
                                   task_type='CPU', verbose=False, thread_count=4)
    print("CatBoost")
    model_fit(model, "CatBoost")


def RGF():
    model = RGFClassifier(max_leaf=100, algorithm="RGF_Sib", verbose=False, n_jobs=-1, min_samples_leaf=10,
                          learning_rate=0.2)
    print("RGF")
    model_fit(model, "RGF")


def RFC():
    model = RandomForestClassifier(n_estimators=1000, max_depth=7, min_samples_leaf=1, max_leaf_nodes=100,
                                   bootstrap=True, n_jobs=-1, warm_start=False)
    print("RFC")
    model_fit(model, "RFC")


def LSVC():
    model_l1 = LinearSVC(penalty='l1', loss='squared_hinge', dual=False, tol=0.001, C=1, multi_class='ovr',
                         fit_intercept=True, verbose=0, max_iter=1000)
    model_l2 = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=0.001, C=1, multi_class='ovr',
                         fit_intercept=True, verbose=0, max_iter=1000)
    # ak mam standardizovane data tak fit_intercept mozem dat na False
    # ak mam vela atributov tak dam dual na True
    print("SVC L1")
    model_fit(model_l1, "SVC_L1")
    print("SVC L2")
    model_fit(model_l2, "SVC_L2")


def SGD():
    model_l1 = SGDClassifier(loss='hinge', penalty='l1', alpha=0.0001, l1_ratio=0.15, max_iter=1000, tol=0.001,
                             shuffle=True, verbose=0, n_jobs=-1, learning_rate='optimal', eta0=0.0, power_t=0.5)
    model_l2 = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, max_iter=1000, tol=0.001,
                             shuffle=True, verbose=0, n_jobs=-1, learning_rate='optimal', eta0=0.0, power_t=0.5)
    model_elastic = SGDClassifier(loss='hinge', penalty='elasticnet', alpha=0.0001, l1_ratio=0.15, max_iter=1000,
                                  tol=0.001, shuffle=True, verbose=0, n_jobs=-1, learning_rate='optimal', eta0=0.0,
                                  power_t=0.5)
    print("SGD SVC L1")
    model_fit(model_l1, "SGD_L1")
    print("SGD SVC, L2")
    model_fit(model_l2, "SGD_L2")
    print("SGD SVC, elasticnet")
    model_fit(model_elastic, "SGD_elasticnet")


def HSIC_lasso(treshold):
    hsic = HSICLasso()
    hsic.input(data, labels)
    before = datetime.datetime.now()
    hsic.classification(treshold, B=19, M=1)
    # B a M su na postupne nacitanie ak mam velky dataset, B deli pocet vzoriek
    after = datetime.datetime.now()
    print("HSIC Lasso")
    selected = hsic.get_index()
    print(len(selected))
    print("cas: " + str(after - before))
    if len(selected) < len(header):
        transform_and_save(selected, "HSIC_Lasso")


def intersections(input_path, output_path):
    files = sorted(glob.glob(input_path))
    names = []
    headers = []
    outputs = []
    for name in files:
        with open(name, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)
            names.append(os.path.basename(f.name)[:-4])
        header = set(header)
        headers.append(header)
    for i in range(len(headers)):
        best = 0
        for j in range(len(headers)):
            if i != j:
                best = max(best, len(headers[i].intersection(headers[j])))
        for k in range(len(headers)):
            if i != k:
                if len(headers[i].intersection(headers[k])) == best:
                    outputs.append(
                        names[i] + " - velkost: " + str(len(headers[i])) + "," + " najlepsi prienik je " + names[
                            k] + ": " + str(len(headers[i].intersection(headers[k]))) + '\n')
    with open(output_path, 'w') as out:
        for output in outputs:
            out.write(output)


data, header, labels = numpy_load()
print("pocet atributov: " + str(len(header)))
print('\n')
percentile = 10
treshold = int(data.shape[1] / 10)  # desatina atributov
# cfs()  # prilis pomaly

# fcbf()
# mifs()
# mrmr()
# cife()
# jmi()
# cmim()
# disr()
# chi_square(percentile)
# MI(percentile)
# f_anova(percentile)
# trace(treshold)
# gini(treshold)
# fisher(treshold)
# lap(treshold)
# xgboost()
# LGBM()
# CAT()
# RGF()
# RFC()
# LSVC()
# SGD()
# HSIC_lasso(treshold)

intersections(output_dir+"*")
sys.stdout.close()
