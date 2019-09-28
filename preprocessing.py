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
new_dir = 'C:/PycharmProjects/Diplomka/skusobny/features/new/'
float_input_dir = 'C:/PycharmProjects/Diplomka/features/tofloat/*'
float_output_dir = 'C:/PycharmProjects/Diplomka/features/float/'
simple_file = 'C:/PycharmProjects/Diplomka/features/simple.csv'
original_file = 'C:/PycharmProjects/Diplomka/features/original.csv'
float_file = 'C:/PycharmProjects/Diplomka/features/float.csv'
features_dir = 'C:/PycharmProjects/Diplomka/features/*'


def normalize(input_path, normal_path):
    files = glob.glob(input_path)
    for name in files:
        print(name)
        with open(name) as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)
            data = list(reader)
        with open(normal_path + os.path.basename(f.name), "w",
                  newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(header)
            scaler = MinMaxScaler(feature_range=(-1, 1), copy=True)
            # normal_data = scaler.fit_transform(data)
            normal_data = np.array(scaler.fit_transform(data), dtype=np.half)  # aby som nemal prilis velku presnost
            writer.writerows(normal_data)


def standardize(input_path, standard_path):
    files = glob.glob(input_path)
    for name in files:
        if os.path.isfile(name):
            with open(name) as f:
                reader = csv.reader(f, delimiter=',')
                header = next(reader)
                data = list(reader)
            with open(standard_path + os.path.basename(f.name), "w",
                      newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow(header)
                scaler = StandardScaler(copy=True)
                # standard_data = scaler.fit_transform(data)
                standard_data = np.array(scaler.fit_transform(data), dtype=np.float32)
                writer.writerows(standard_data)


# def half_float(input_path, new_path):
#     files = sorted(glob.glob(input_path))
#     for name in files:
#         with open(name) as f:
#             reader = csv.reader(f, delimiter=',')
#             header = next(reader)
#             data = list(reader)
#             with open(new_path + os.path.basename(f.name), "w", newline='') as csv_file:
#                 writer = csv.writer(csv_file, delimiter=',')
#                 writer.writerow(header)
#                 writer.writerows(np.array(data, dtype=np.half))


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


def merge_features(input_path, output_file):  # zluci atributy z jednotlivych suborov do jedneho
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
    # oddelenie float atributov, robim pre disassembled, instructions, overlay, strings, sections, potom aj normal grams
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
                frame[column] = frame[column].astype('int32')
        frame.drop(columns=colummns_to_drop, inplace=True)
        frame.to_csv(output_dir + os.path.basename(file), header=True, index=False)
    float_frame.to_csv(output_dir + "float.csv", header=True, index=False)


# float_hotfix(float_input_dir, float_output_dir)
# divide(original_path, simple_dir)
# merge_features(original_path, original_file)
# merge_features(simple_dir+'*', simple_file)
# merge_features(float_output_dir+'*', float_file)
# merged_sizes(original_file, simple_file, float_file)
# standardize(features_dir, standard_dir)
