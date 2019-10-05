import glob
import os
import csv
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
import sys

original_path = 'C:/PycharmProjects/Diplomka/features/original/*'
normal_dir = 'C:/PycharmProjects/Diplomka/features/normal/'
standard_dir = 'C:/PycharmProjects/Diplomka/features/standard/'
simple_dir = 'C:/PycharmProjects/Diplomka/features/simple/'
new_dir = 'C:/PycharmProjects/Diplomka/features/new/'
float_input_dir = 'C:/PycharmProjects/Diplomka/features/tofloat/*'
float_output_dir = 'C:/PycharmProjects/Diplomka/features/float/'
simple_file = 'C:/PycharmProjects/Diplomka/features/simple.csv'
original_file = 'C:/PycharmProjects/Diplomka/features/original.csv'
float_file = 'C:/PycharmProjects/Diplomka/features/float.csv'
discrete_file = 'C:/PycharmProjects/Diplomka/features/discrete.csv'
simple_discrete_file = 'C:/PycharmProjects/Diplomka/features/simple_discrete.csv'
original_discrete_file = 'C:/PycharmProjects/Diplomka/features/original_discrete.csv'
features_dir = 'C:/PycharmProjects/Diplomka/features/*'


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
            # normal_data = np.array(scaler.fit_transform(data), dtype=np.half)  # aby som nemal prilis velku presnost
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
                # standard_data = np.array(scaler.fit_transform(data), dtype=np.float32)
                writer.writerows(standard_data)


def divide(input_path, simple_path):  # jednoduche atributy sa skopiruju do ineho priecinka
    files = glob.glob(input_path)
    for name in files:
        with open(name) as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)
            if len(header) < 1000:
                data = list(reader)
                f.close()
                with open(simple_path + os.path.basename(f.name), "w", newline='') as csv_file:
                    writer = csv.writer(csv_file, delimiter=',')
                    writer.writerow(header)
                    writer.writerows(data)
        # else:
        #     with open(complex_path + os.path.basename(f.name), "w", newline='') as csv_file:
        #         writer = csv.writer(csv_file, delimiter=',')
        #         writer.writerow(header)
        #         writer.writerows(data)


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


def merged_sizes(original_file, simple_file, float_file):
    with open(original_file) as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        print(len(header))
    with open(simple_file) as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        print(len(header))
    with open(float_file) as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        print(len(header))


def float_hotfix(input_dir, output_dir):  # robim aj pretypovanie na float32
    # oddelenie float atributov
    # robim pre disassembled, instructions, overlay, strings, sections, potom aj normal grams ak bude treba
    # pred zmergovanim float pridam cele sizes a file_entropy
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


def discretize(input_file, new_file, decimals):
    # obmedzi pocet cislic za desatinnou ciarkou - x, vynasobi 10^x a pretypuje na int
    with open(input_file) as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        data = list(reader)
    max_decimals = [0]*len(header)  # max pocet desat. cislic v string tvare atributov
    for line in data:
        for i in range(len(line)):
            max_decimals[i] = max(max_decimals[i], (len(line[i]) - line[i].index('.') - 1))
    for i in range(len(max_decimals)):
        if max_decimals[i] == 1:
            max_decimals[i] = 0  # len jedno cislo ignorujem, casto to je len nula
        if max_decimals[i] > decimals:
            max_decimals[i] = decimals
    with open(new_file, "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(header)
        data = np.array(data, dtype=np.float64)
        for i in range(len(max_decimals)):
            data[:, i] = np.around(data[:, i], decimals=max_decimals[i])
            data[:, i] = data[:, i]*np.power(10, max_decimals[i])
        writer.writerows(data.astype(np.uint64))


def prefix_hotfix(input_dir, output_dir):  # prefixy budu uz celym nazvom atributu podla mena suboru
    files = glob.glob(input_dir)
    for name in files:
        data = np.loadtxt(name, delimiter=',', skiprows=1, dtype=np.uint64)
        if type(data[0]) is not np.uint64:
            header = [os.path.basename(name)[:-4]]*len(data[0])
        else:  # niektore skupiny maju len jeden atribut
            header = [os.path.basename(name)[:-4]]
        with open(output_dir + os.path.basename(name), "w", newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(header)
            if type(data[0]) is not np.uint64:
                writer.writerows(data)
            else:
                writer.writerows([i] for i in data)


# float_hotfix(float_input_dir, float_output_dir)  # najprv odddelim float atributy
# divide(original_path, simple_dir)  # potom odddelim jednoduche
# merge_features_from_dir(simple_dir+'*', simple_file)
# merge_features_from_dir(original_path, original_file)
# merge_features_from_dir(float_output_dir+'*', float_file)
# discretize(float_file, discrete_file, 5)  # diskretizujem float a ulozim ho do original
# merge_feature_files(discrete_file, simple_file, simple_discrete_file)  # zmergujem discrete a simple
# merge_feature_files(discrete_file, original_file, original_discrete_file)  # zmergujem discrete a original
# merged_sizes(original_file, simple_file, float_file)
# na konci vsetky atributy standardizujem, predtym vymazem povodne subory, ostanu len zmergovane !!!
# standardize(features_dir, standard_dir)

# prefix_hotfix(original_path, new_dir)
# prefix_hotfix(simple_dir + "*", new_dir)
