import pandas as pd
import numpy as np
import glob
import os
import csv
import collections
import sys
import re

best_features_path = 'features/selection/*'
intersections_file = 'vysledky/intersections.txt'
best_groups_output_file = 'vysledky/groups.txt'
simple_file = 'features/simple.csv'
very_simple_file = 'features/very_simple.csv'
original_file = 'features/original.csv'
selected_results = 'vysledky/selected/*'
compact_selected_results = 'vysledky/compact_selected/'


def intersections(input_path, output_path):
    # pocet atributov v ktorych sa prelinaju. Ked som presiel na menovanie skupinami uz to nepouzivam
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
        if best != 0:
            for k in range(len(headers)):
                if i != k:
                    if len(headers[i].intersection(headers[k])) == best:
                        outputs.append(
                            names[i] + " - velkost: " + str(len(headers[i])) + "," + " najlepsi prienik je " + names[
                                k] + ": " + str(len(headers[i].intersection(headers[k]))) + '\n')
    with open(output_path, 'w') as out:
        for output in outputs:
            out.write(output)


def best_groups(input_dir, output_file):
    features = set(np.loadtxt(original_file, delimiter=',', max_rows=1, dtype="str"))
    simple_features = set(np.loadtxt(simple_file, delimiter=',', max_rows=1, dtype="str"))
    very_simple_features = set(np.loadtxt(very_simple_file, delimiter=',', max_rows=1, dtype="str"))
    files = sorted(glob.glob(input_dir))
    with open(output_file, 'w') as out:
        for file in files:
            groups = collections.Counter()
            simple = 0
            very_simple = 0
            header = np.loadtxt(file, delimiter=',', max_rows=1, dtype="str")
            for i in range(len(header)):
                for feature in features:
                    if header[i] == feature or header[i] == ('"' + feature + '"'):
                        groups[feature] += 1
                        if feature in simple_features:
                            simple += 1
                            if feature in very_simple_features:
                                very_simple += 1
            out.write(os.path.basename(file)[:-4] + '\n')
            out.write("pocet vybranych: " + str(len(header)) + '\n')
            out.write(str(groups.most_common()) + '\n')
            out.write("simple: " + str(simple) + '\n')
            out.write("very simple: " + str(very_simple) + '\n')
            out.write('\n')


def result_processing(input_dir, output_dir):  # kompaktnejsie spracovanie "selected" vysledkov
    files = glob.glob(input_dir)
    results = []
    for file_name in files:
        with open(file_name, "r") as result_file:
            with open(output_dir + os.path.basename(file_name), "w") as output_file:
                try:
                    while True:
                        output_file.write(next(result_file))
                        next(result_file)
                        while True:
                            line = next(result_file)
                            splitted = re.split('\s', line)
                            # print(splitted)
                            results.append(splitted[2])
                            next_line = "  "
                            while next_line != '\n':
                                next_line = next(result_file)  # su tam dve prazdne riadky
                            next(result_file)
                            if next(result_file) == "------------------------------------------------------------\n":
                                break
                        output_file.write(str(min(results)) + " " + str(max(results)) + '\n')
                        next(result_file)
                        next(result_file)
                        results = []
                except StopIteration:
                    continue


# intersections(best_features_path, intersections_file)
# best_groups(best_features_path, best_groups_output_file)
# result_processing(selected_results, compact_selected_results)
