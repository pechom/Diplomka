import pandas as pd
import numpy as np
import glob
import os
import csv
import collections
import sys

# potom urobim priecinok kde dam najlepsie hodnotene selekcie atributov. na nich budem robit prieniky aj skupiny
best_features_path = 'seminar/selection/*'
features_path = 'seminar/selection/*'
intersections_file = 'seminar/intersections.txt'
best_groups_output_file = 'seminar/groups.txt'


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


def best_groups(input_dir, output_file):  # pomen polia skupin podla novych nazvov
    features = ["export", "import_libs", "import_funcs", "metadata", "overlay", "sections", "resources",
                "strings", "byte_entropy_histogram", "sizes", "file_entropy", "instructions", "disassembled",
                "1-gram_bin_strings", "2-gram_char_freq_strings", "2-gram_normal_freq_dis_hex",
                "2-gram_bin_strings", "1-gram_freq_strings", "2-gram_freq_strings", "1-gram_bin_hex", "2-gram_bin_hex",
                "1-gram_freq_hex", "2-gram_freq_hex", "1-gram_normal_freq_hex", "2-gram_normal_freq_hex",
                "1-gram_char_bin_strings", "2-gram_char_bin_strings", "1-gram_char_freq_strings",
                "1-gram_opcodes", "2-gram_opcodes", "1-gram_freq_opcodes", "2-gram_freq_opcodes",
                "1-gram_reg", "2-gram_reg", "1-gram_freq_reg", "2-gram_freq_reg", "1-gram_dis_hex", "2-gram_dis_hex",
                "1-gram_freq_dis_hex", "2-gram_freq_dis_hex", "1-gram_normal_freq_dis_hex"]
    simple_features = ["1-gram_bin_hex", "1-gram_bin_strings", "2-gram_bin_strings",
                       "1-gram_dis_hex", "1-gram_freq_dis_hex", "1-gram_normal_freq_dis_hex", "1-gram_freq_hex",
                       "1-gram_normal_freq_hex", "1-gram_freq_reg", "1-gram_freq_opcodes", "1-gram_char_bin_strings",
                       "2-gram_char_bin_strings", "1-gram_char_freq_strings", "2-gram_char_freq_strings",
                       "byte_entropy_histogram", "export", "import_libs", "import_funcs", "instructions",
                       "disassembled", "metadata", "overlay", "sections", "resources", "strings",
                       "file_entropy", "sizes"]  # posledne dve su z float
    files = glob.glob(input_dir)
    with open(output_file, 'w') as out:
        for file in files:
            groups = collections.Counter()
            simple = 0
            float = 0
            header = np.loadtxt(file, delimiter=',', max_rows=1, dtype="str")
            for i in range(len(header)):
                for feature in features:
                    if header[i].startswith(feature):
                        groups[feature] += 1
                for feature in simple_features:
                    if header[i].startswith(feature) or header[i].startswith('"' + feature):
                        simple += 1
            out.write(os.path.basename(file)[:-4] + '\n')
            out.write("pocet vybranych: " + str(len(header)) + '\n')
            out.write(str(groups.most_common()) + '\n')
            out.write("simple: " + str(simple) + '\n')
            out.write('\n')


# intersections(features_path, intersections_file)
best_groups(best_features_path, best_groups_output_file)

# with open('C:/PycharmProjects/Diplomka/skusobny/classification/3-gram_bin_strings.csv') as ff:
#     reader = csv.reader(ff, delimiter=',')
#     header = next(reader)
#     for head in header:
#         print(head)
