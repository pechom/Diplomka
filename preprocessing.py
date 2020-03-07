import glob
import csv
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
import collections
import os
import shutil

original_path = 'features/original/*'
standard_dir = 'features/standard/'
simple_dir = 'features/simple/'
very_simple_dir = 'features/very_simple/'
simple_file = 'features/simple.csv'
very_simple_file = 'features/very_simple.csv'
original_file = 'features/original.csv'
features_dir = 'features/*'
discrete_dir = 'discrete/'
labels_path = 'subory/labels.csv'

discretize_decimals = 5  # pocet desatinnych miest na ktore diskretizujem
simple_treshold = 1000  # max pocet atributov skupiny ktora bude patrit medzi jednoduche skupiny
very_simple_treshold = 100  # max pocet atributov skupiny ktora bude patrit medzi velmi jednoduche skupiny


def normalize(input_path, normal_path):
    files = glob.glob(input_path)
    for name in files:
        with open(name) as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)
            data = list(reader)
        with open(normal_path + os.path.basename(f.name), "w", newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(header)
            scaler = MinMaxScaler(feature_range=(-1, 1), copy=True)
            normal_data = scaler.fit_transform(data)
            writer.writerows(normal_data)


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


def divide_simple(input_path, simple_path, treshold):  # jednoduche atributy sa skopiruju do ineho priecinka
    files = glob.glob(input_path)
    for name in files:
        with open(name) as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)
            if len(header) <= treshold:
                data = list(reader)
                with open(simple_path + os.path.basename(f.name), "w", newline='') as csv_file:
                    writer = csv.writer(csv_file, delimiter=',')
                    writer.writerow(header)
                    writer.writerows(data)


def merge_features_from_dir(input_path, output_file):  # zluci atributy z priecinka do jedneho
    files = glob.glob(input_path)
    alldata = []
    for name in files:
        with open(name) as f:
            reader = csv.reader(f)
            data = list(reader)
        if not alldata:
            alldata = data
        else:
            for i in range(len(alldata)):
                alldata[i].extend(data[i])
    with open(output_file, "w", newline='') as f2:
        writer = csv.writer(f2)
        writer.writerows(alldata)


def merge_feature_files(file_to_merge_1, file_to_merge_2, output_file):
    with open(file_to_merge_1) as f:
        reader = csv.reader(f)
        data = list(reader)
        alldata = data
    with open(file_to_merge_2) as f:
        reader = csv.reader(f)
        data = list(reader)
    for i in range(len(alldata)):
        alldata[i].extend(data[i])
    with open(output_file, "w", newline='') as f2:
        writer = csv.writer(f2)
        writer.writerows(alldata)


def feature_size(file):
    with open(file) as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        counter = collections.Counter(header)
        print(os.path.basename(file)[:-4])
        print(counter)
        print(len(header))


def float_hotfix(input_dir, output_dir):
    # oddelenie float atributov. Ked som presiel na diskretizaciu to uz nepouzivam
    files = glob.glob(input_dir)
    float_frame = pd.DataFrame()
    column_loc = 0
    for file in files:
        frame = pd.read_csv(file)
        colummns_to_drop = []
        for column in frame.columns.values:
            if "/" in column or "average" in column or "entropy" in column:
                float_frame.insert(column_loc, column=column, value=frame[column])
                float_frame[column] = float_frame[column].astype('float32')
                column_loc += 1
                colummns_to_drop.append(column)
            else:
                frame[column] = frame[column].astype('int64')
        frame.drop(columns=colummns_to_drop, inplace=True)
        frame.to_csv(output_dir + os.path.basename(file), header=True, index=False)
    float_frame.to_csv(output_dir + "float.csv", header=True, index=False)


def discretize(input_path, output_dir, decimals):
    # obmedzi pocet cislic za desatinnou ciarkou - x, vynasobi 10^x a pretypuje na int
    files = glob.glob(input_path)
    for feature_file in files:
        with open(feature_file) as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)
            data = list(reader)
        max_decimals = [0] * len(header)  # max pocet desat. cislic v string tvare atributov
        for line in data:
            for i in range(len(line)):
                float_position = line[i].find('.')
                if float_position != -1:  # pri celych cislach necham nulu
                    max_decimals[i] = max(max_decimals[i], (len(line[i]) - float_position - 1))
        #         absolutna lebo find da -1 ked nenajde
        for i in range(len(max_decimals)):
            if max_decimals[i] <= 1:
                max_decimals[i] = 0  # len jedno cislo ignorujem, casto to je len nula
            if max_decimals[i] > decimals:
                max_decimals[i] = decimals
        with open(output_dir + os.path.basename(feature_file), "w", newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(header)
            data = np.array(data, dtype=np.float64)
            for i in range(len(max_decimals)):
                data[:, i] = np.around(data[:, i], decimals=max_decimals[i])
                data[:, i] = data[:, i] * np.power(10, max_decimals[i])
            writer.writerows(data.astype(np.uint64))


def prefix_hotfix(input_dir, output_dir):
    # prefixy budu uz celym nazvom atributu podla mena suboru. Ked som presiel na menovanie skupinami to uz nepouzivam.
    files = glob.glob(input_dir)
    for name in files:
        data = np.loadtxt(name, delimiter=',', skiprows=1, dtype=np.uint64)
        if type(data[0]) is not np.uint64:
            header = [os.path.basename(name)[:-4]] * len(data[0])
        else:  # niektore skupiny maju len jeden atribut
            header = [os.path.basename(name)[:-4]]
        with open(output_dir + os.path.basename(name), "w", newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(header)
            if type(data[0]) is not np.uint64:
                writer.writerows(data)
            else:
                writer.writerows([i] for i in data)


def create_dataset_from_clusters(input_dir, output_dir, clusters_file, new_labels_file, number_type):
    # odstranim labels aj atributy pre vzorky ktore su outliere
    # uz to nepouzivam, odstranovanie outlierov som presunul do tvorby labels kvoli optimalizacii
    cluster_labels = np.loadtxt(clusters_file, delimiter=',', skiprows=1, dtype=np.int8)
    to_delete = []
    for i in range(len(cluster_labels)):
        if cluster_labels[i] == -1:  # outlier
            to_delete.append(i)
    labels = np.delete(cluster_labels, to_delete)
    with open(new_labels_file, "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerows(([i] for i in labels))
    files = glob.glob(input_dir)
    for file_name in files:
        if os.path.isfile(file_name):
            header = np.loadtxt(file_name, delimiter=',', max_rows=1, dtype="str")
            data = np.loadtxt(file_name, delimiter=',', skiprows=1, dtype=number_type)
            data = np.delete(data, to_delete, axis=0)
            with open(output_dir + os.path.basename(file_name), "w", newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow(header)
                writer.writerows(data)


# -------------------------------------------------------

def main():
    os.mkdir(discrete_dir)
    # diskretizujem, potom pôvodne zahodim a novy dir premenujem na original
    discretize(features_dir, discrete_dir, discretize_decimals)
    shutil.rmtree(features_dir[:-1])
    os.renames(discrete_dir, original_path[:-2])
    os.mkdir(simple_dir)
    os.mkdir(very_simple_dir)
    divide_simple(original_path, simple_dir, simple_treshold)  # potom odddelim jednoduche
    divide_simple(original_path, very_simple_dir, very_simple_treshold)  # odddelim velmi jednoduche
    merge_features_from_dir(simple_dir + '*', simple_file)
    merge_features_from_dir(original_path, original_file)
    merge_features_from_dir(very_simple_dir + '*', very_simple_file)
    for name in [original_file, simple_file, very_simple_file]:
        feature_size(name)
    # vymazem povodne subory, ostanu len zmergovane
    shutil.rmtree(original_path[:-1])
    shutil.rmtree(simple_dir)
    shutil.rmtree(very_simple_dir)
    os.mkdir(standard_dir)
    # na konci vsetky atributy standardizujem
    standardize(features_dir, standard_dir)


if __name__ == "__main__":
    main()
