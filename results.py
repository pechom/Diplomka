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
sys.stdout = open('seminar/groups.txt', 'w')


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


def best_groups(input_dir, output_dir):
    features = ["export_", "import_lib_", "import_func_", "metadata_", "overlay_", "sections_", "resources_",
                "strings_", "histogram_", "sizes_", "file_entropy", "instructions_", "disassembled_", "bin_strings_",
                "freq_strings_", "bin_hex_", "freq_hex_", "freq_hex_normal_", "char_bin_strings_", "char_freq_strings_",
                "hex_opcode_", "freq_hex_opcode_", "opcodes_", "freq_opcodes_", "reg_", "freq_reg_", "dis_hex_",
                "freq_dis_hex_", "freq_dis_hex_normal_"]
    simple_features = ["bin_hex_", "bin_strings_", "dis_hex_", "freq_dis_hex_", "freq_dis_hex_normal_", "freq_hex_",
                       "freq_hex_normal_", "freq_reg_", "freq_opcodes_", "char_bin_strings_", "char_freq_strings_",
                       "histogram_", "export_", "import_lib_", "import_func_", "instructions_", "disassembled_",
                       "metadata_", "overlay_", "sections_", "resources_", "strings_",
                       "file_entropy", "sizes_"]  # posledne dve su z float
    float_features = ["file_entropy", "disassembled_average_length", "instructions_all/allocation",
                      "instructions_all/jump", "overlay_entropy", "sections_file_size/size_of_known",
                      "sections_file_size/size_of_unknown", "sections_size_of_known/unknown",
                      "sections_size_of_unknown/known", "sections_.text_entropy", "sections_.text_size/file_size",
                      "sections_.data_entropy", "sections_.data_size/file_size", "sections_.bss_size/file_size",
                      "sections_.rdata_entropy", "sections_.rdata_size/file_size", "sections_.idata_entropy",
                      "sections_.idata_size/file_size", "sections_.rsrc_entropy", "sections_.rsrc_size/file_size",
                      "sections_.tls_size/file_size", "sections_.reloc_entropy", "sections_.reloc_size/file_size",
                      "strings_average_length", "strings_entropy", "sizes_file_size", "sizes_hex_size",
                      "sizes_disassembled_size", "sizes_file/disassembled_size", "sizes_hex/disassembled_size",
                      "sizes_disassembled/file_size", "sizes_disassembled/hex_size"]
    files = glob.glob(input_dir)
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
                if header[i].startswith(feature):
                    simple += 1
        print(os.path.basename(file)[:-4])
        print(groups.most_common())
        print("simple: " + str(simple))
        print('\n')


# intersections(features_path, intersections_file)
best_groups(best_features_path, best_groups_output_file)

# with open('C:/PycharmProjects/Diplomka/skusobny/classification/3-gram_bin_strings.csv') as ff:
#     reader = csv.reader(ff, delimiter=',')
#     header = next(reader)
#     for head in header:
#         print(head)

sys.stdout.close()
