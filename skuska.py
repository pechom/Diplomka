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

X, y = datasets.load_iris(return_X_y=True)

# ---------------
# ukladanie pre sklearn
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

# ----------------

# ukladanie lgbm
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
# --------------------
# model = lgb.LGBMClassifier(max_depth=7, learning_rate=0.2, n_estimators=100, objective='multiclass',
#                                      n_jobs=-1, num_leaves=80, min_child_samples=10, importance_type='split',
#                                      num_class=10)
# model.fit(X, y)
# with open('lgb.pickle', 'wb') as handle:
#     pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('lgb.pickle', 'rb') as handle:
#     model2 = pickle.load(handle)
# z = model2.predict(X[0:1])
# print(z)

# ----------

# ukladanie xgboost
# dtrain = xgb.DMatrix(data=X, label=y)
# param = {'max_depth': 7, 'objective': 'multi:softmax', 'eval_metric': 'merror', 'num_class': 10,
#          'learning_rate': 0.2, 'n_jobs': -1, 'min_child_weight': 10}
# result = xgb.train(param, dtrain, num_boost_round=50)
# result.save_model("xgb.txt")
# bst = xgb.Booster()
# bst.load_model('xgb.txt')
# z = bst.predict(xgb.DMatrix(X[0:1]))
# print(z)

# ----
# ukladanie scalera - na predikovanom datasete robim len transform (musi mat rovnake parametre)
# sc = StandardScaler()
# X_train_std = sc.fit_transform(X_train)
# pickle.dump(sc, open('file/path/scaler.pkl','wb'))
# sc = pickle.load(open('file/path/scaler.pkl','rb'))
