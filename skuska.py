from skfeature.function.statistical_based import gini_index
from skfeature.function.similarity_based import SPEC, fisher_score, reliefF, trace_ratio, lap_score
import datetime
import pandas as pd
import numpy as np
import glob
import csv
import os
import shutil

# feature_path = 'C:/PycharmProjects/Diplomka/skusobny/classification/3-gram_bin_strings.csv'
# standard_feature_path = 'C:/PycharmProjects/Diplomka/skusobny/classification/standard_3-gram_bin_strings.csv'
# labels_path = 'C:/PycharmProjects/Diplomka/skusobny/classification/clear_labels2_head.csv'
# output_dir = "C:/PycharmProjects/Diplomka/skusobny/selection/"
#
#
# def gini(treshold):
#     before = datetime.datetime.now()
#     result = gini_index.gini_index(data, labels, mode="index")
#     after = datetime.datetime.now()
#     print("Gini")
#     result = result[:treshold]
#     print(len(result))
#     print("cas: " + str(after - before))
#     print('\n')
#
#
# def fisher(treshold):
#     before = datetime.datetime.now()
#     result = fisher_score.fisher_score(data, labels, mode="index")
#     after = datetime.datetime.now()
#     print("fisher")
#     result = result[:treshold]
#     print(len(result))
#     print("cas: " + str(after - before))
#     print('\n')
#
#
# def lap(treshold):
#     before = datetime.datetime.now()
#     result = lap_score.lap_score(data, mode="index")
#     after = datetime.datetime.now()
#     print("lap")
#     result = result[:treshold]
#     print(len(result))
#     print("cas: " + str(after - before))
#     print('\n')
#
#
# def trace(treshold):
#     before = datetime.datetime.now()
#     result = trace_ratio.trace_ratio(data, labels, mode="index")
#     after = datetime.datetime.now()
#     print("trace")
#     result = result[:treshold]
#     print(len(result))
#     print("cas: " + str(after - before))
#     print('\n')
#
#
# def numpy_load():
#     labels = np.loadtxt(labels_path, delimiter=',', skiprows=1, dtype=np.int8)
#     data = np.loadtxt(feature_path, delimiter=',', skiprows=1, dtype=np.int32)
#     return data, labels
#
#
# data, labels = numpy_load()
# trace(treshold)
# gini(treshold)
# fisher(treshold)
# lap(treshold)

# pole = np.array([[0.1, 0.01], [0.005, 0.00004], [0.0007, 210.0], [25.0, 48.0], [0.001, 0.002]])
# # print(pole)
# for i in range(5):
#     pole[i, :] = np.around(pole[i, :])
#     print(pole[i, :])


# print(pole)
# print(np.around(pole))
# print(np.array(pole*np.power(10, 3), dtype=np.int))
# features='C:/PycharmProjects/Diplomka/skuska/features/*'
# discrete='C:/PycharmProjects/Diplomka/skuska/discrete/'
# original = 'C:/PycharmProjects/Diplomka/skuska/features/original/*'
#
# # shutil.rmtree(features[:-1])
# os.renames(discrete[:-1], original[:-2])

# very_simple_file = 'C:/PycharmProjects/Diplomka/features/very_simple.csv'
# print(set(np.loadtxt(very_simple_file, delimiter=',', max_rows=1, dtype="str")))
