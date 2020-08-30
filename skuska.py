from sklearn import svm
from sklearn import datasets
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import xgboost as xgb
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import csv
import json
import collections
import pprint
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import nltk

#
# X, y = datasets.load_iris(return_X_y=True)
#
# # ---------------
# # ukladanie pre sklearn
# clf = svm.SVC()
# clf.fit(X, y)
# with open('svc.pickle', 'wb') as handle:
#     pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('svc.pickle', 'rb') as handle:
#     clf2 = pickle.load(handle)
# z = clf2.predict(X)
# print(z)
# mat = confusion_matrix(y, z)  # triedy su zoradene ciselne
# names = ("Iris Setosa", "Iris Versicolour", "Iris Virginica")  # z class_number
# sn.heatmap(mat, annot=True, annot_kws={"size": 18}, xticklabels=names, yticklabels=names)
# plt.show()
# plt.savefig('subory/plot.png')
#
# # ----------------
#
# # ukladanie lgbm
# param = {"max_depth": 7, "learning_rate": 0.2, "objective": 'multiclass', 'num_leaves': 80,
#          "num_class": 3, "metric": 'multi_error', 'min_data_in_leaf': 10, 'num_threads': -1, 'verbosity': -1,
#          'min_data_in_bin': 3, 'max_bin': 255, 'enable_bundle': True, 'max_conflict_rate': 0.0}
# dtrain = lgb.Dataset(data=X, label=y)
# model = lgb.train(param, dtrain, num_boost_round=50, verbose_eval=None)  # cv nevracia model, len vysledky
# # model.save_model('lgb.txt', num_iteration=model.best_iteration)
# # model = lgb.Booster(model_file="lgb.txt")
# z = model.predict(X)  # vracia pravdepodobnosti pre kazdu triedu
# res = []
# for i in range(len(z)):
#     a = (np.where(z == np.max(z[i]))[1])[0]
#     res.append(a)
# print(res)
# # --------------------
# model = lgb.LGBMClassifier(max_depth=7, learning_rate=0.2, n_estimators=100, objective='multiclass',
#                            n_jobs=-1, num_leaves=80, min_child_samples=10, importance_type='split',
#                            num_class=10)
# model.fit(X, y)
# with open('lgb.pickle', 'wb') as handle:
#     pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('lgb.pickle', 'rb') as handle:
#     model2 = pickle.load(handle)
# z = model2.predict(X[0:1])
# print(z)
#
# # ----------
#
# # ukladanie xgboost
# dtrain = xgb.DMatrix(data=X, label=y)
# param = {'max_depth': 7, 'objective': 'multi:softmax', 'eval_metric': 'merror', 'num_class': 10,
#          'learning_rate': 0.2, 'n_jobs': -1, 'min_child_weight': 10}
# result = xgb.train(param, dtrain, num_boost_round=50)
# result.save_model("xgb.txt")
# bst = xgb.Booster()
# bst.load_model('xgb.txt')
# z = bst.predict(xgb.DMatrix(X[0:1]))
# print(z)
#
# # ----
# # ukladanie scalera - na predikovanom datasete robim len transform (musi mat rovnake parametre)
# sc = StandardScaler()
# X_train_std = sc.fit_transform(X)
# pickle.dump(sc, open('file/path/scaler.pkl', 'wb'))
# sc = pickle.load(open('file/path/scaler.pkl', 'rb'))
#
# CONFIG_FILE = 'vt-config.json'
# CONFIG_FOLDER = 'D:/praca/'
# filepath = CONFIG_FOLDER + CONFIG_FILE
# init_conf = {
#     'api_key': '0c4047d034a7f8e0f5afd8884d47af625df14618c8ec5c98ff03d599a8f5441e',
#     'api_url': 'https://www.virustotal.com/vtapi/v2/',
#     'output': 'json'
# }
# f = open(filepath, 'w')
# f.write(json.dumps(init_conf))
# f.close()
#
# feature_path = 'features/original.csv'
# header = np.loadtxt(feature_path, delimiter=',', max_rows=1, dtype="str")
# print(len(header))
# print(header)
#
# subor = "skuska.txt"
# with open(subor) as original:
#     line = original.readline()
#     print(len(line))
#     print(line)
#
# with open(subor) as f:
#     reader = csv.reader(f, delimiter=',')
#     line = next(reader)
#     print(len(line))
#     print(line)
#
# header = np.loadtxt(subor, delimiter=',', max_rows=1, dtype="str")
# print(len(header))
# print(header)
#
#
# def word_clearing(word):
#     for ch in ['\\', ',', '\'', '\"', '\a', '\b', '\f', '\n', '\r', '\t', '\v', '\0', '\1', '\2', '\3', '\4', '\5',
#                '\6', '\7']:
#         if ch in word:
#             word = word.replace(ch, '?')
#     return word
#
#
# def header_clearing(header):
#     for i in range(len(header)):
#         for ch in ['\\', ',', '\'', '\"', '\a', '\b', '\f', '\n', '\r', '\t', '\v', '\0', '\1', '\2', '\3', '\4', '\5',
#                    '\6', '\7']:
#             if ch in header[i]:
#                 header[i] = header[i].replace(ch, '?')
#     return header
#
#
# name = "report.json"
# with open(name) as f:
#     data = json.load(f)
# print(data["additional_info"]["pe-resource-types"]['RT_ICON'])
# for type in data["additional_info"]["pe-resource-types"]:
#     print(type)
# func = word_clearing(data["additional_info"]["imports"]['MPR.dll'])
# print(func)
# lib = header_clearing(data["additional_info"]["imports"]['MPR.dll'])
# print(lib)
#
# header = ["bla_alb", "aaa_bbb"]
# header = np.array(header)
# header = [x[:x.index("_")] for x in header]
# print(header)
#
# data = np.loadtxt("skuska.txt", delimiter=',', skiprows=1, dtype=np.uint64)
#
# sel = VarianceThreshold()
# try:
#     sel.fit_transform(data)
#     support = sel.get_support(True)
#     print("number of irrelevant features: " + str(len(data[0]) - len(support)))
# except ValueError:
#     print("empty")

# text = "just some text to test \n and more text"
# n = 2
# tokenized1 = text.split()
# tokenized2 = list(text)
# string_grams = list(nltk.ngrams(tokenized1, n))
# char_grams = list(nltk.ngrams(tokenized2, n))
# print(string_grams)
# print(char_grams)
# char_grams = [text[i:i + n] for i in range(len(text) - n + 1)]
# string_grams = [tokenized1[i:i + n] for i in range(len(tokenized1) - n + 1)]
# string_grams = [' '.join(grams) for grams in string_grams]
# print(string_grams)
# print(char_grams)
