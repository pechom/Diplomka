from sklearn import svm
from sklearn import datasets
import pickle
import xgboost as xgb
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
X, y = datasets.load_iris(return_X_y=True)


# ---------------
# ukladanie pre sklearn
# clf = svm.SVC()
# clf.fit(X, y)
# with open('svc.pickle', 'wb') as handle:
#     pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('svc.pickle', 'rb') as handle:
#     clf2 = pickle.load(handle)
# z = clf2.predict(X[0:1])
# print(z)

# ----------------

# ukladanie lgbm
# param = {"max_depth": 7, "learning_rate": 0.2, "objective": 'multiclass', 'num_leaves': 80,
#          "num_class": 10, "metric": 'multi_error', 'min_data_in_leaf': 10, 'num_threads': -1, 'verbosity': -1,
#          'min_data_in_bin': 3, 'max_bin': 255, 'enable_bundle': True, 'max_conflict_rate': 0.0}
# dtrain = lgb.Dataset(data=X, label=y)
# result = lgb.train(param, dtrain, num_boost_round=50, verbose_eval=None)  # cv nevracia model, len vysledky
# result.save_model('lgb.txt', num_iteration=result.best_iteration)
# model = lgb.Booster(model_file="lgb.txt")
# z = model.predict(X[0:1], num_iteration=50)  # vracia pravdepodobnosti pre kazdu triedu
# print(((np.where(z == np.max(z)))[0])[0])  # pre kazdy riadok, potom to pridat do pola
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

subor = 'subory/labels3.csv'
with open(subor) as f:
    print(os.path.basename(subor)[:-4])
