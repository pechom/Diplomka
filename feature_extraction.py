import json
import collections
import glob
import csv
import os
import time
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import nltk
import math
import sys
import re
import preprocessing
import classification
import shutil

reports_path = 'reports/*'
string_path = 'strings/*'
hex_path = 'hex/*'
disassembled_path = 'disassembled/*'
entropy_file = 'subory/entropy.csv'
opcodes_path = 'disassembled_divided/opcodes/*'
registers_path = 'disassembled_divided/registers/*'
instructions_path = 'disassembled_divided/instructions/*'
headers_dir = 'features/headers/'

df_max_count = 10  # maximalny pocet vyskytov pre DF pri ktorom odstranim atribut
max_ngram = 2  # maximalne n pre ktore robim n-gram
for_prediction = False  # ci robim predikciu
first_time = False  # pre divide disassembled
selected_file = ''


def document_frequency_selection(counter):
    for name, count in counter.copy().items():
        if count < df_max_count + 1:
            del counter[name]
    return counter


def variance_treshold_selection(data):
    treshold = 0.05
    # if len(data[0]) >= 1000:
    #     treshold = 0.05
    # if len(data[0]) >= 10000:
    #     treshold = 0.1
    sel = VarianceThreshold(threshold=treshold)
    try:
        data = sel.fit_transform(data)
    except ValueError:
        selected = []
        for i in range(len(data[0])):
            selected.append(i)
        return [], selected
    return data, sel.get_support(True)  # indexy vybranych


def entropy(string):
    # get probability of chars in string
    prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
    # calculate the entropy
    entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
    return entropy


def features_to_csv(header, data, name):
    if not (len(data) == 0 or len(header) == 0):
        for i in range(len(header)):
            header[i] = name + "_" + header[i]
        with open("features/" + name + ".csv", "w", newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(header)
            writer.writerows(data)
        print("finalna dlzka " + str(len(header)))
    else:
        print("finalna dlzka" + " 0")


def header_from_selection(selected, variance_selected, message):
    header = []
    if len(selected) != 0:
        for i in range(len(selected)):
            if i in variance_selected:
                header.append(selected[i])
        if len(selected) in variance_selected:
            if message != "":
                header.append(message)
        header = header_clearing(header)
    return header


def delete_not_selected(features, header):
    features, selected = variance_treshold_selection(features)
    for i in reversed(range(len(header))):
        if i not in selected:
            del header[i]
    return features, header


def word_clearing(word):
    for ch in ['\\', ',', '\'', '\"', '\a', '\b', '\f', '\n', '\r', '\t', '\v', '\0', '\1', '\2', '\3', '\4', '\5',
               '\6', '\7']:
        if ch in word:
            word = word.replace(ch, '?')
    return word


def header_clearing(header):
    for i in range(len(header)):
        for ch in ['\\', ',', '\'', '\"', '\a', '\b', '\f', '\n', '\r', '\t', '\v', '\0', '\1', '\2', '\3', '\4', '\5',
                   '\6', '\7']:
            if ch in header[i]:
                header[i] = header[i].replace(ch, '?')
    return header


def clear_prefix_from_header(prefix):  # vyberiem z hlavicky len atributy vybranej skupiny a odsranim prefix
    prefix_header = np.loadtxt(selected_file, delimiter=',', max_rows=1, dtype="str")
    header = []
    for header_name in prefix_header:
        if header_name.startswith(prefix):
            header.append(header_name[len(prefix):])
    return header


def create_import_libs_features(path, prefix):
    if not for_prediction:
        libraries = collections.Counter()
        files = sorted(glob.glob(path))
        for name in files:
            with open(name) as f:
                data = json.load(f)
            for imp in data["additional_info"]["imports"]:
                libraries[imp] += 1
        print(len(libraries))
        libraries = document_frequency_selection(libraries)
        print("po DF " + str(len(libraries)))
        header = []
        features = []
        if len(libraries) > 0:
            selected_libs = list(libraries.keys())
            for name in files:
                with open(name) as f:
                    data = json.load(f)
                feature = [0] * (len(selected_libs) + 1)
                for i in range(len(selected_libs)):
                    for imp in data["additional_info"]["imports"]:
                        if selected_libs[i] in imp:
                            feature[i] = len(data["additional_info"]["imports"][selected_libs[i]])  # pocet funkcii
                feature[-1] = len(data["additional_info"]["imports"])  # pocet kniznic
                features.append(feature)
            features, selected = variance_treshold_selection(features)
            header = header_from_selection(selected_libs, selected, "number_of_DLLs")
    else:
        header = clear_prefix_from_header(prefix)
        features = []
        if len(header) != 0:
            files = sorted(glob.glob(path))
            for name in files:
                with open(name) as f:
                    data = json.load(f)
                feature = [0] * (len(header))
                imps = []
                for imp in data["additional_info"]["imports"]:
                    imps.append(imp)
                imps = header_clearing(imps)
                for i in range(len(header)):
                    for lib in imps:
                        if header[i] in lib:
                            feature[i] = len(data["additional_info"]["imports"][header[i]])  # pocet funkcii
                if "number_of_DLLs" in header[-1]:
                    feature[-1] = len(data["additional_info"]["imports"])  # pocet kniznic
                features.append(feature)
    return header, features


def create_import_func_features(path, prefix):
    if not for_prediction:
        functions = collections.Counter()
        files = sorted(glob.glob(path))
        for name in files:
            with open(name) as f:
                data = json.load(f)
            for imp in data["additional_info"]["imports"]:
                for func in data["additional_info"]["imports"][imp]:
                    functions[func] += 1
        print(len(functions))
        functions = document_frequency_selection(functions)
        print("po DF " + str(len(functions)))
        features = []
        header = []
        if len(functions) != 0:
            selected_funcs = list(functions.keys())
            for name in files:
                with open(name) as f:
                    data = json.load(f)
                feature = [0] * (len(selected_funcs))
                for i in range(len(selected_funcs)):
                    for imp in data["additional_info"]["imports"]:
                        if selected_funcs[i] in data["additional_info"]["imports"][imp]:
                            feature[i] = 1  # vyskyt funkcie
                features.append(feature)
            features, selected = variance_treshold_selection(features)
            header = header_from_selection(selected_funcs, selected, "")
    else:
        header = clear_prefix_from_header(prefix)
        features = []
        if len(header) != 0:
            files = sorted(glob.glob(path))
            for name in files:
                with open(name) as f:
                    data = json.load(f)
                feature = [0] * (len(header))
                for i in range(len(header)):
                    for imp in data["additional_info"]["imports"]:
                        funcs = header_clearing(data["additional_info"]["imports"][imp])
                        for func in funcs:
                            if header[i] in func:
                                feature[i] = 1
                features.append(feature)
    return header, features


def create_export_features(path, prefix):
    if not for_prediction:
        libraries = collections.Counter()
        files = sorted(glob.glob(path))
        for name in files:
            with open(name) as f:
                data = json.load(f)
            if "exports" in data["additional_info"]:
                for imp in data["additional_info"]["exports"]:
                    libraries[imp] += 1
        print(len(libraries))
        libraries = document_frequency_selection(libraries)
        print("po DF " + str(len(libraries)))
        header = []
        features = []
        if len(libraries) > 0:
            selected_libs = list(libraries.keys())
            for name in files:
                with open(name) as f:
                    data = json.load(f)
                feature = [0] * (len(selected_libs) + 1)
                if "exports" in data["additional_info"]:
                    for i in range(len(selected_libs)):
                        if selected_libs[i] in data["additional_info"]["exports"]:
                            feature[i] = 1  # vyskyt exportu
                    feature[-1] = len(data["additional_info"]["exports"])  # pocet exportov
                features.append(feature)
            features, selected = variance_treshold_selection(features)
            header = header_from_selection(selected_libs, selected, "number_of_DLLs")
    else:
        header = clear_prefix_from_header(prefix)
        features = []
        if len(header) != 0:
            files = sorted(glob.glob(path))
            for name in files:
                with open(name) as f:
                    data = json.load(f)
                feature = [0] * (len(header))
                if "exports" in data["additional_info"]:
                    for i in range(len(header)):
                        libs = header_clearing(data["additional_info"]["exports"])
                        for lib in libs:
                            if header[i] in lib:
                                feature[i] = 1
                    if "number_of_DLLs" in header[-1]:
                        feature[-1] = len(data["additional_info"]["exports"])
                    features.append(feature)
    return header, features


def create_metadata_features(path, prefix):
    if not for_prediction:
        files = sorted(glob.glob(path))
        features = []
        header = ["CodeSize", "LinkerVersion", "InitializedDataSize", "UninitializedDataSize",
                  "OSVersion", "ImageVersion", "TimeStamp", "EntryPoint", "SubsystemVersion", "MachineType", "Size"]
        print(len(header))
        for name in files:
            with open(name) as f:
                data = json.load(f)
            feature = [0] * 11
            feature[0] = data["additional_info"]["exiftool"]["CodeSize"]
            feature[1] = data["additional_info"]["exiftool"]["LinkerVersion"]
            feature[2] = data["additional_info"]["exiftool"]["InitializedDataSize"]
            feature[3] = data["additional_info"]["exiftool"]["UninitializedDataSize"]
            feature[4] = data["additional_info"]["exiftool"]["OSVersion"]
            feature[5] = data["additional_info"]["exiftool"]["ImageVersion"]
            dt = data["additional_info"]["exiftool"]["TimeStamp"]
            if dt[:19] == "0000:00:00 00:00:00":
                feature[6] = 0
            else:
                dt = time.strptime(dt[:19], '%Y:%m:%d %X')
                feature[6] = int(time.mktime(dt))
            # feature[6] = data["additional_info"]["pe-timestamp"]
            # feature[7] = int(data["additional_info"]["exiftool"]["EntryPoint"])
            feature[7] = data["additional_info"]["pe-entry-point"]
            feature[8] = data["additional_info"]["exiftool"]["SubsystemVersion"]
            feature[9] = data["additional_info"]["pe-machine-type"]
            feature[10] = data["size"]
            features.append(feature)
        features, header = delete_not_selected(features, header)
    else:
        header = clear_prefix_from_header(prefix)
        features = []
        if len(header) != 0:
            files = sorted(glob.glob(path))
            feature_names = ["CodeSize", "LinkerVersion", "InitializedDataSize", "UninitializedDataSize",
                             "OSVersion", "ImageVersion", "TimeStamp", "EntryPoint", "SubsystemVersion", "MachineType",
                             "Size"]
            for name in files:
                with open(name) as f:
                    data = json.load(f)
                feature = [0] * (len(header))
                counter = 0
                if feature_names[0] in header:
                    feature[counter] = data["additional_info"]["exiftool"]["CodeSize"]
                    counter += 1
                if feature_names[1] in header:
                    feature[counter] = data["additional_info"]["exiftool"]["LinkerVersion"]
                    counter += 1
                if feature_names[2] in header:
                    feature[counter] = data["additional_info"]["exiftool"]["InitializedDataSize"]
                    counter += 1
                if feature_names[3] in header:
                    feature[counter] = data["additional_info"]["exiftool"]["UninitializedDataSize"]
                    counter += 1
                if feature_names[4] in header:
                    feature[counter] = data["additional_info"]["exiftool"]["OSVersion"]
                    counter += 1
                if feature_names[5] in header:
                    feature[counter] = data["additional_info"]["exiftool"]["ImageVersion"]
                    counter += 1
                if feature_names[6] in header:
                    dt = data["additional_info"]["exiftool"]["TimeStamp"]
                    if dt[:19] == "0000:00:00 00:00:00":
                        feature[counter] = 0
                    else:
                        dt = time.strptime(dt[:19], '%Y:%m:%d %X')
                        feature[counter] = int(time.mktime(dt))
                    counter += 1
                if feature_names[7] in header:
                    feature[counter] = data["additional_info"]["pe-entry-point"]
                    counter += 1
                if feature_names[8] in header:
                    feature[counter] = data["additional_info"]["exiftool"]["SubsystemVersion"]
                    counter += 1
                if feature_names[9] in header:
                    feature[counter] = data["additional_info"]["pe-machine-type"]
                    counter += 1
                if feature_names[10] in header:
                    feature[counter] = data["size"]
                features.append(feature)
    return header, features


def create_overlay_features(path, prefix):
    if not for_prediction:
        files = sorted(glob.glob(path))
        features = []
        header = ["entropy", "offset", "size"]
        print(len(header))
        for name in files:
            with open(name) as f:
                data = json.load(f)
            feature = [0] * 3
            if "pe-overlay" in data["additional_info"]:
                feature[0] = data["additional_info"]["pe-overlay"]["entropy"]
                feature[1] = data["additional_info"]["pe-overlay"]["offset"]
                feature[2] = data["additional_info"]["pe-overlay"]["size"]
            features.append(feature)
        features, header = delete_not_selected(features, header)
    else:
        header = clear_prefix_from_header(prefix)
        features = []
        if len(header) != 0:
            files = sorted(glob.glob(path))
            feature_names = ["entropy", "offset", "size"]
            for name in files:
                with open(name) as f:
                    data = json.load(f)
                    feature = [0] * (len(header))
                if "pe-overlay" in data["additional_info"]:
                    counter = 0
                    if feature_names[0] in header:
                        feature[counter] = data["additional_info"]["pe-overlay"]["entropy"]
                        counter += 1
                    if feature_names[1] in header:
                        feature[counter] = data["additional_info"]["pe-overlay"]["offset"]
                        counter += 1
                    if feature_names[2] in header:
                        feature[counter] = data["additional_info"]["pe-overlay"]["size"]
                features.append(feature)
    return header, features


def create_section_features(path, prefix):
    known_sections = [".text", ".data", ".bss", ".rdata", ".edata", ".idata", ".rsrc", ".tls", ".reloc"]
    header = ["number_of_known", "number_of_unknown", "number_of_all", "size_of_known", "size_of_unknown",
              "file_size/size_of_known", "file_size/size_of_unknown", "number_of_empty_size", "number_of_empty_name",
              "size_of_known/unknown", "size_of_unknown/known"]
    initial_size = len(header)
    for section in known_sections:
        header.append(section + "_virtual_address")
        header.append(section + "_physical_address")
        header.append(section + "_size")
        header.append(section + "_entropy")
        header.append(section + "_size/file_size")
    if not for_prediction:
        files = sorted(glob.glob(path))
        features = []
        for name in files:
            with open(name) as f:
                data = json.load(f)
            feature = [0] * (len(header) + (9 * 5))
            file_size = data["size"]
            number_of_known = 0
            number_of_unknown = 0
            size_of_known = 0
            size_of_unknown = 0
            number_of_empty_size = 0
            number_of_empty_name = 0
            section_index = 0
            for section in data["additional_info"]["sections"]:
                if section[0] in known_sections:
                    for i in range(len(known_sections)):
                        if known_sections[i] == section[0]:
                            section_index = i
                            break
                    iterator = initial_size + 5 * section_index
                    number_of_known += 1
                    feature[iterator] = section[1]
                    iterator += 1
                    feature[iterator] = section[2]
                    iterator += 1
                    feature[iterator] = section[3]
                    size_of_known += feature[iterator]
                    if feature[iterator] == 0:
                        number_of_empty_size += 1
                    iterator += 1
                    feature[iterator] = section[4]
                    iterator += 1
                    if section[3] != 0:
                        feature[iterator] = file_size / section[3]
                    else:
                        feature[iterator] = file_size
                    iterator += 1
                else:
                    if len(section[0]) == 0:
                        number_of_empty_name += 1
                    if section[3] == 0:
                        number_of_empty_size += 1
                    number_of_unknown += 1
                    size_of_unknown += section[3]
            feature[0] = number_of_known
            feature[1] = number_of_unknown
            feature[2] = number_of_known + number_of_unknown
            feature[3] = size_of_known
            feature[4] = size_of_unknown
            if size_of_known != 0:
                feature[5] = file_size / size_of_known
                feature[10] = size_of_unknown / size_of_known
            else:
                feature[5] = file_size
                feature[10] = size_of_unknown
            if size_of_unknown != 0:
                feature[6] = file_size / size_of_unknown
                feature[9] = size_of_known / size_of_unknown
            else:
                feature[6] = file_size
                feature[9] = size_of_known
            feature[7] = number_of_empty_size
            feature[8] = number_of_empty_name
            features.append(feature)
        features, header = delete_not_selected(features, header)
    else:
        header = clear_prefix_from_header(prefix)
        features = []
        if len(header) != 0:
            files = sorted(glob.glob(path))
            for name in files:
                with open(name) as f:
                    data = json.load(f)
                    feature = [0] * (len(header))
                header_start = ["number_of_known", "number_of_unknown", "number_of_all", "size_of_known",
                                "size_of_unknown",
                                "file_size/size_of_known", "file_size/size_of_unknown", "number_of_empty_size",
                                "number_of_empty_name",
                                "size_of_known/unknown", "size_of_unknown/known"]
                file_size = data["size"]
                number_of_known = 0
                number_of_unknown = 0
                size_of_known = 0
                size_of_unknown = 0
                number_of_empty_size = 0
                number_of_empty_name = 0
                for section in data["additional_info"]["sections"]:
                    if section[0] in known_sections:
                        number_of_known += 1
                        if (section[0] + "_virtual_address") in header:
                            feature[header.index(section[0] + "_virtual_address")] = section[1]
                        if (section[0] + "_physical_address") in header:
                            feature[header.index(section[0] + "_physical_address")] = section[2]
                        if (section[0] + "size") in header:
                            feature[header.index(section[0] + "_size")] = section[3]
                        size_of_known += section[3]
                        if section[3] == 0:
                            number_of_empty_size += 1
                        if (section[0] + "_entropy") in header:
                            feature[header.index(section[0] + "_entropy")] = section[4]
                        if section[3] != 0:
                            if (section[0] + "_entropy") in header:
                                feature[header.index(section[0] + "_entropy")] = file_size / section[3]
                        else:
                            if (section[0] + "_entropy") in header:
                                feature[header.index(section[0] + "_entropy")] = file_size
                    else:
                        if len(section[0]) == 0:
                            number_of_empty_name += 1
                        if section[3] == 0:
                            number_of_empty_size += 1
                        number_of_unknown += 1
                        size_of_unknown += section[3]
                counter = 0
                if header_start[0] in header:
                    feature[counter] = number_of_known
                    counter += 1
                if header_start[1] in header:
                    feature[counter] = number_of_unknown
                    counter += 1
                if header_start[2] in header:
                    feature[counter] = number_of_unknown + number_of_unknown
                    counter += 1
                if header_start[3] in header:
                    feature[counter] = number_of_known
                    counter += 1
                if header_start[4] in header:
                    feature[counter] = number_of_unknown
                    counter += 1
                if size_of_known != 0:
                    if header_start[5] in header:
                        feature[counter] = file_size / size_of_known
                        counter += 1
                else:
                    if header_start[5] in header:
                        feature[counter] = file_size
                        counter += 1
                if size_of_unknown != 0:
                    if header_start[6] in header:
                        feature[counter] = file_size / size_of_unknown
                        counter += 1
                else:
                    if header_start[6] in header:
                        feature[counter] = file_size / size_of_unknown
                        counter += 1
                if header_start[7] in header:
                    feature[counter] = number_of_empty_size
                    counter += 1
                if header_start[8] in header:
                    feature[counter] = number_of_empty_name
                    counter += 1
                if size_of_unknown != 0:
                    if header_start[9] in header:
                        feature[counter] = size_of_known / size_of_unknown
                        counter += 1
                else:
                    if header_start[9] in header:
                        feature[counter] = size_of_known
                        counter += 1
                if size_of_known != 0:
                    if header_start[10] in header:
                        feature[counter] = size_of_unknown / size_of_known
                else:
                    if header_start[10] in header:
                        feature[counter] = size_of_unknown
                features.append(feature)
    return header, features


def create_resource_features(path, prefix):
    if not for_prediction:
        resource_types = collections.Counter()
        files = sorted(glob.glob(path))
        for name in files:
            with open(name) as f:
                data = json.load(f)
            if "pe-resource-types" in data["additional_info"]:
                for resource_type in data["additional_info"]["pe-resource-types"]:
                    resource_types[resource_type] += 1
        print(len(resource_types))
        resource_types = document_frequency_selection(resource_types)
        print("po DF " + str(len(resource_types)))
        header = []
        features = []
        if len(resource_types) > 0:
            all_resources = 0
            selected_types = list(resource_types.keys())
            for name in files:
                with open(name) as f:
                    data = json.load(f)
                feature = [0] * (len(selected_types) + 1)
                for i in range(len(selected_types)):
                    if "pe-resource-types" in data["additional_info"]:
                        if selected_types[i] in data["additional_info"]["pe-resource-types"]:
                            feature[i] = data["additional_info"]["pe-resource-types"][selected_types[i]]
                            all_resources = 0
                            for resource_type in data["additional_info"]["pe-resource-types"]:
                                all_resources += data["additional_info"]["pe-resource-types"][resource_type]
                    feature[-1] = all_resources
                features.append(feature)
            features, selected = variance_treshold_selection(features)
            header = header_from_selection(selected_types, selected, "number_of_resources")
    else:
        header = clear_prefix_from_header(prefix)
        features = []
        if len(header) != 0:
            files = sorted(glob.glob(path))
            for name in files:
                with open(name) as f:
                    data = json.load(f)
                feature = [0] * (len(header))
                if "pe-resource-types" in data["additional_info"]:
                    types = []
                    for type in data["additional_info"]["pe-resource-types"]:
                        types.append(type)
                    types = header_clearing(types)
                    for i in range(len(header)):
                        for type in types:
                            if header[i] in type:
                                feature[i] = data["additional_info"]["pe-resource-types"][type]
                    if "number_of_resources" in header[-1]:
                        all_resources = 0
                        for resource_type in data["additional_info"]["pe-resource-types"]:
                            all_resources += data["additional_info"]["pe-resource-types"][resource_type]
                        feature[-1] = all_resources
                features.append(feature)
    return header, features


def create_string_features(path, prefix):
    max_length = 0
    bins = []
    header = []
    features = []
    for i in range(5, 10, 1):
        bins.append(i)
    for i in range(10, 100, 10):
        bins.append(i)
    for i in range(100, 1000, 100):
        bins.append(i)
    for i in range(1000, 10001, 1000):
        bins.append(i)
    bins.append(sys.maxsize)
    if not for_prediction:
        for i in range(len(bins) - 1):
            header.append(str(bins[i]) + "-" + str(bins[i + 1]))
        header.extend(
            ["C", "http", "HKEY", "MZ", "IP", "average_length", "max_length", "number_of_strings", "file_size",
             "entropy"])
        files = sorted(glob.glob(path))
        for name in files:
            feature = []
            line_lengths = []
            with open(name) as f:
                text = f.readlines()
                for line in text:
                    line_lengths.append(len(line))
                    max_length = max(max_length, len(line))
                hist = np.histogram(line_lengths, bins=bins)
                histogram = hist[0]
                feature.extend(histogram)
                f.seek(0)
                text = f.read()
            low_text = text.lower()
            feature.append(low_text.count("c:\\\\"))
            feature.append(low_text.count("http"))
            feature.append(low_text.count("hkey"))
            feature.append(text.count("mz"))
            feature.append(len(re.findall(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", text)))
            feature.append(np.mean(line_lengths))
            feature.append(max_length)
            feature.append(len(line_lengths))
            feature.append(os.path.getsize(f.name))
            feature.append(entropy(text))
            features.append(feature)
        features, header = delete_not_selected(features, header)
    else:
        header = clear_prefix_from_header(prefix)
        features = []
        if len(header) != 0:
            files = sorted(glob.glob(path))
            for name in files:
                feature = [0] * (len(header))
                line_lengths = []
                with open(name) as f:
                    text = f.readlines()
                    for line in text:
                        line_lengths.append(len(line))
                        max_length = max(max_length, len(line))
                    hist = np.histogram(line_lengths, bins=bins)
                    histogram = hist[0]
                    f.seek(0)
                    text = f.read()
                low_text = text.lower()
                for i in range(len(bins) - 1):
                    if (str(bins[i]) + "-" + str(bins[i + 1])) in header:
                        feature[header.index(str(bins[i]) + "-" + str(bins[i + 1]))] = histogram[i]
                next_header = ["C", "http", "HKEY", "MZ", "IP", "average_length", "max_length", "number_of_strings",
                               "file_size", "entropy"]
                if next_header[0] in header:
                    feature[header.index(next_header[0])] = low_text.count("c:\\\\")
                if next_header[1] in header:
                    feature[header.index(next_header[1])] = low_text.count("http")
                if next_header[2] in header:
                    feature[header.index(next_header[2])] = low_text.count("hkey")
                if next_header[3] in header:
                    feature[header.index(next_header[3])] = low_text.count("mz")
                if next_header[4] in header:
                    feature[header.index(next_header[4])] = len(
                        re.findall(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", text))
                if next_header[5] in header:
                    feature[header.index(next_header[5])] = np.mean(line_lengths)
                if next_header[6] in header:
                    feature[header.index(next_header[6])] = max_length
                if next_header[7] in header:
                    feature[header.index(next_header[7])] = len(line_lengths)
                if next_header[8] in header:
                    feature[header.index(next_header[8])] = os.path.getsize(f.name)
                if next_header[9] in header:
                    feature[header.index(next_header[9])] = entropy(text)
                features.append(feature)
    return header, features


def entropy_bin_counts(block, window):
    # source https://arxiv.org/pdf/1508.03096.pdf
    c = np.bincount(block >> 4, minlength=16)
    p = c.astype(np.float32) / window
    wh = np.where(c)[0]
    h = np.sum(-p[wh] * np.log2(
        p[wh])) * 2
    hbin = int(h * 2)
    if hbin == 16:
        hbin = 15
    return hbin, c


def byte_entropy_histogram(bytez, step, window):
    # source https://arxiv.org/pdf/1508.03096.pdf
    output = np.zeros((16, 16), dtype=np.int)
    a = np.frombuffer(bytez, dtype=np.uint8)
    if a.shape[0] < window:
        hbin, c = entropy_bin_counts(a, window)
        output[hbin, :] += c
    else:
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step, :]
        for block in blocks:
            hbin, c = entropy_bin_counts(block, window)
            output[hbin, :] += c
    return output.flatten().tolist()


def create_byte_entropy_histogram_features(path, prefix, step, window):
    if not for_prediction:
        files = sorted(glob.glob(path))
        counters = []
        global_counter = collections.Counter()
        features = []
        header = []
        for name in files:
            with open(name) as f:
                hexes = f.read()
            bytez = bytes.fromhex(hexes)
            histogram = byte_entropy_histogram(bytez, step, window)
            counter = collections.Counter()
            for i in range(len(histogram)):
                if counter[i] == 0:
                    counter[i] = 1
            counters.append(counter)
        for i in range(len(counters)):
            global_counter = global_counter + counters[i]
        print(len(global_counter))  # ma byt 256
        global_counter = document_frequency_selection(global_counter)
        print("po DF " + str(len(global_counter)))
        if len(global_counter) > 0:
            selected_indices = list(global_counter.keys())
            for name in files:
                with open(name) as f:
                    hexes = f.read()
                bytez = bytes.fromhex(hexes)
                feature = byte_entropy_histogram(bytez, step, window)
                for i in reversed(range(len(feature))):
                    if i not in selected_indices:
                        del feature[i]
                features.append(feature)
            features, selected = variance_treshold_selection(features)
            for i in range(len(selected_indices)):
                if i in selected:
                    header.append((str(selected_indices[i])))
    else:
        header = clear_prefix_from_header(prefix)
        features = []
        if len(header) != 0:
            files = sorted(glob.glob(path))
            for name in files:
                with open(name) as f:
                    hexes = f.read()
                bytez = bytes.fromhex(hexes)
                feature = byte_entropy_histogram(bytez, step, window)
                selected_feature = []
                for i in range(len(feature)):
                    if str(i) in header:
                        selected_feature.append(feature[i])
                features.append(selected_feature)
    return header, features


def create_sizes_features(reports_path, hex_path, disassembled_path, prefix):
    header = ["file_size", "hex_size", "disassembled_size", "file/hex_size", "file/disassembled_size", "hex/file_size",
              "hex/disassembled_size", "disassembled/file_size", "disassembled/hex_size"]
    feature_names = header.copy()
    if not for_prediction:
        files = sorted(glob.glob(reports_path))
        features = []
        print(len(header))
        for name in files:
            with open(name) as f:
                data = json.load(f)
                basename = os.path.basename(f.name)
            feature = [0] * 11
            feature[0] = data["size"]
            feature[1] = os.path.getsize(hex_path[:-1] + basename)
            feature[2] = os.path.getsize(disassembled_path[:-1] + basename)
            feature[3] = feature[0] / feature[1]
            feature[4] = feature[0] / feature[2]
            feature[5] = feature[1] / feature[0]
            feature[6] = feature[1] / feature[2]
            feature[7] = feature[2] / feature[0]
            feature[8] = feature[2] / feature[1]
            features.append(feature)
        features, header = delete_not_selected(features, header)
    else:
        header = clear_prefix_from_header(prefix)
        features = []
        feature_data = []
        if len(header) != 0:
            files = sorted(glob.glob(reports_path))
            for name in files:
                with open(name) as f:
                    data = json.load(f)
                    basename = os.path.basename(f.name)
                feature = [0] * (len(header))
                feature_data.append(data["size"])
                feature_data.append(os.path.getsize(hex_path[:-1] + basename))
                feature_data.append(os.path.getsize(disassembled_path[:-1] + basename))
                counter = 0
                if feature_names[0] in header:
                    feature[counter] = feature_data[0]
                    counter += 1
                if feature_names[1] in header:
                    feature[counter] = feature_data[1]
                    counter += 1
                if feature_names[2] in header:
                    feature[counter] = feature_data[2]
                    counter += 1
                if feature_names[3] in header:
                    feature[counter] = feature_data[0] / feature_data[1]
                    counter += 1
                if feature_names[4] in header:
                    feature[counter] = feature_data[0] / feature_data[2]
                    counter += 1
                if feature_names[5] in header:
                    feature[counter] = feature_data[1] / feature_data[0]
                    counter += 1
                if feature_names[6] in header:
                    feature[counter] = feature_data[1] / feature_data[2]
                    counter += 1
                if feature_names[7] in header:
                    feature[counter] = feature_data[2] / feature_data[0]
                    counter += 1
                if feature_names[8] in header:
                    feature[counter] = feature_data[2] / feature_data[1]
                features.append(feature)
    return header, features


def create_entropy_feature(file, prefix):
    if not for_prediction:
        header = ["file_entropy"]
    else:
        header = clear_prefix_from_header(prefix)
    features = []
    if len(header) != 0:
        with open(file) as f:
            lines = f.readlines()
        for line in lines:  # subory su rovnako usporiadane ako ked prechadzam priecinok cez glob, testoval som to
            feature = [0]
            feature[0] = line.split()[1]
            features.append(feature)
    return header, features


def divide_disassembled_files(disassembled_path, opcodes_path, registers_path, instructions_path):
    files = sorted(glob.glob(disassembled_path))
    for name in files:
        with open(name, errors='replace') as f:
            basename = os.path.basename(f.name)
            text = f.readlines()
        del text[0:7]  # v prvych riadkoch nie su instrukcie
        with open(opcodes_path[:-1] + basename, 'w') as f, open(instructions_path[:-1] + basename, 'w') as f2, open(
                registers_path[:-1] + basename, 'w') as f3:
            for line in text:
                if not ((line == "\n") or ("section" in line) or (">:" in line) or ("..." in line) or (
                        "Disassembly" in line)):
                    tokens = line.split("\t")
                    del tokens[0]  # umiestnenie riadku (poradie) v povodnom subore
                    if len(tokens) > 0:
                        hexes = tokens[0].replace(" ", "")
                        hexes = hexes.replace("\n", "")
                        f.write(hexes + " ")
                    if len(tokens) > 1:
                        opcode = tokens[1].split()[0]
                        opcode = opcode.replace("\n", "")
                        f2.write(opcode + " ")
                    for token in tokens:
                        token = re.split('[, ]', token)
                        for secondary_token in token:
                            if "%" in secondary_token:
                                start_index = secondary_token.index("%")
                                secondary_token = secondary_token.replace("\n", "")
                                f3.write(secondary_token[start_index + 1:start_index + 4] + " ")


def create_instruction_features(opcodes_path, dis_hex_path, registers_path, prefix):
    header = ["number_of_allocation", "number_of_jump", "all/allocation",
              "all/jump", "all", "hex", "registers"]
    feature_names = header.copy()
    allocation_instructions = ["dd", "db", "dw", "dup"]
    jump_instructions = ["jmp", "je", "jne", "jg", "jge", "ja", "jae", "jl", "jle", "jb", "jbe", "jo", "jno", "jz",
                         "jnz", "js", "jns", "jcxz", "jecxz", "jrcxz"]
    if not for_prediction:
        files = sorted(glob.glob(opcodes_path))
        features = []
        print(len(header))
        for name in files:
            feature = [0] * len(header)
            with open(name) as f:
                basename = os.path.basename(f.name)
                instructions = f.read().split()
            with open(dis_hex_path[:-1] + basename) as f2, open(registers_path[:-1] + basename) as f3:
                hex_text = f2.read().split()
                reg_text = f3.read().split()
            number_of_instructions = len(instructions)
            for instruction in instructions:
                if instruction in allocation_instructions:
                    feature[0] += 1
                else:
                    if instruction in jump_instructions:
                        feature[1] += 1
            if feature[0] != 0:
                feature[2] = number_of_instructions / feature[0]
            else:
                feature[2] = number_of_instructions
            if feature[1] != 0:
                feature[3] = number_of_instructions / feature[1]
            else:
                feature[3] = number_of_instructions
            feature[4] = number_of_instructions
            feature[5] = len(hex_text)
            feature[6] = len(reg_text)
            features.append(feature)
        features, header = delete_not_selected(features, header)
    else:
        header = clear_prefix_from_header(prefix)
        features = []
        if len(header) != 0:
            files = sorted(glob.glob(path))
            for name in files:
                feature = [0] * len(header)
                feature_help = [0] * 2
                counter = 0
                with open(name) as f:
                    basename = os.path.basename(f.name)
                    instructions = f.read().split()
                with open(dis_hex_path[:-1] + basename) as f2, open(registers_path[:-1] + basename) as f3:
                    hex_text = f2.read().split()
                    reg_text = f3.read().split()
                number_of_instructions = len(instructions)
                for instruction in instructions:
                    if instruction in allocation_instructions:
                        feature_help[0] += 1
                    else:
                        if instruction in jump_instructions:
                            feature_help[1] += 1
                if feature_names[0] in header:
                    feature[counter] = feature_help[0]
                    counter += 1
                if feature_names[1] in header:
                    feature[counter] = feature_help[1]
                    counter += 1
                if feature_help[0] != 0:
                    if feature_names[2] in header:
                        feature[counter] = number_of_instructions / feature[0]
                        counter += 1
                else:
                    if feature_names[2] in header:
                        feature[counter] = number_of_instructions
                        counter += 1
                if feature_help[1] != 0:
                    if feature_names[3] in header:
                        feature[counter] = number_of_instructions / feature[1]
                        counter += 1
                else:
                    if feature_names[3] in header:
                        feature[counter] = number_of_instructions
                        counter += 1
                if feature_names[4] in header:
                    feature[counter] = number_of_instructions
                    counter += 1
                if feature_names[5] in header:
                    feature[counter] = len(hex_text)
                    counter += 1
                if feature_names[6] in header:
                    feature[counter] = len(reg_text)
                features.append(feature)
    return header, features


def create_disassembled_features(path, prefix):
    if not for_prediction:
        files = sorted(glob.glob(path))
        sizes = collections.Counter()
        for name in files:
            with open(name, errors='replace') as f:
                text = f.readlines()
            for line in text:
                sizes[len(line)] += 1
        print(len(sizes))
        sizes = document_frequency_selection(sizes)
        print("po DF " + str(len(sizes)))
        header = []
        features = []
        if len(sizes) > 0:
            selected_sizes = list(sizes.keys())
            for name in files:
                line_lengths = []
                feature = [0] * (len(selected_sizes) + 2)
                with open(name, errors='replace') as f:
                    text = f.readlines()
                for line in text:
                    length = len(line)
                    line_lengths.append(length)
                    if length in selected_sizes:
                        feature[selected_sizes.index(length)] += 1
                    # max_length = max(max_length, len(line))
                feature[len(selected_sizes)] = np.mean(line_lengths)
                feature[len(selected_sizes) + 1] = len(line_lengths)
                features.append(feature)
            # print(max_length)
            features, selected = variance_treshold_selection(features)
            for i in range(len(selected_sizes)):
                if i in selected:
                    header.append(str(selected_sizes[i]))
            if len(selected_sizes) in selected:
                header.append("average_length")
            if (len(selected_sizes) + 1) in selected:
                header.append("number_of_lines")
    else:
        header = clear_prefix_from_header(prefix)
        features = []
        if len(header) != 0:
            files = sorted(glob.glob(path))
            for name in files:
                line_lengths = []
                feature = [0] * (len(header))
                with open(name, errors='replace') as f:
                    text = f.readlines()
                for line in text:
                    length = len(line)
                    line_lengths.append(length)
                    if str(length) in header:
                        feature[header.index(str(length))] += 1
                if "average_length" in header:
                    feature[header.index("average_length")] = np.mean(line_lengths)
                if "number_of_lines" in header:
                    feature[header.index("number_of_lines")] = len(line_lengths)
                features.append(feature)
    return header, features


def create_n_grams(path, n, is_char, bin_prefix, freq_prefix):
    if not for_prediction:
        files = sorted(glob.glob(path))
        counters = []
        global_counter = collections.Counter()
        for name in files:
            with open(name) as f:
                text = f.read()
            if not is_char:
                tokenized = text.split()  # split rozdeli na slova
            else:
                tokenized = list(text)  # list rozdeli na pismena
            grams = list(nltk.ngrams(tokenized, n))
            counter = collections.Counter()
            for gram in grams:
                if counter[gram] == 0:
                    counter[gram] = 1
            counters.append(counter)
        for i in range(len(counters)):
            global_counter = global_counter + counters[i]
        print(len(global_counter))
        global_counter = document_frequency_selection(global_counter)
        print("po DF " + str(len(global_counter)))
        bin_header = []
        freq_header = []
        bin_features = []
        freq_features = []
        if len(global_counter) > 0:
            selected_grams = list(global_counter.keys())
            for name in files:
                with open(name) as f:
                    text = f.read()
                if not is_char:
                    tokenized = text.split()
                else:
                    tokenized = list(text)
                grams = list(nltk.ngrams(tokenized, n))
                grams_freq = collections.Counter(grams)
                bin_feature = [0] * len(selected_grams)
                freq_feature = [0] * len(selected_grams)
                for i in range(len(selected_grams)):
                    if grams_freq[selected_grams[i]] != 0:
                        bin_feature[i] = 1
                        freq_feature[i] = grams_freq[selected_grams[i]]
                bin_features.append(bin_feature)
                freq_features.append(freq_feature)
            bin_features, bin_selected = variance_treshold_selection(bin_features)
            freq_features, freq_selected = variance_treshold_selection(freq_features)
            bin_header = header_from_selection(selected_grams, bin_selected, "")
            freq_header = header_from_selection(selected_grams, freq_selected, "")
    else:
        bin_header = clear_prefix_from_header(bin_prefix)
        freq_header = clear_prefix_from_header(freq_prefix)
        bin_features = []
        freq_features = []
        files = sorted(glob.glob(path))
        for name in files:
            with open(name) as f:
                text = f.read()
            if not is_char:
                tokenized = text.split()
            else:
                tokenized = list(text)
            grams = list(nltk.ngrams(tokenized, n))
            grams = header_clearing(grams)
            grams_freq = collections.Counter(grams)
            bin_feature = [0] * len(bin_header)
            for i in range(len(bin_header)):
                for j in range(len(grams)):
                    if bin_header[i] in grams[j]:
                        if grams_freq[grams[j]] != 0:
                            bin_feature[i] = 1
            freq_feature = [0] * len(freq_header)
            for i in range(len(freq_header)):
                for gram in grams_freq:
                    if freq_header[i] in gram:
                        freq_feature[i] = grams_freq[gram]
            bin_features.append(bin_feature)
            freq_features.append(freq_feature)
    return bin_header, bin_features, freq_header, freq_features


def create_hex_grams(path, n, bin_prefix, freq_prefix, normal_freq_prefix):
    if not for_prediction:
        files = sorted(glob.glob(path))
        counters = []
        global_counter = collections.Counter()
        for name in files:
            with open(name) as f:
                text = f.read()
                text = text.replace(" ", "")
                # dvojice pismen su jeden byte - 1-gram
            grams = [text[i:i + 2 * n] for i in range(0, (len(text) - 2 * n + 1), 2)]
            counter = collections.Counter()
            for gram in grams:
                if counter[gram] == 0:
                    counter[gram] = 1
            counters.append(counter)
        for i in range(len(counters)):
            global_counter = global_counter + counters[i]
        print(len(global_counter))
        global_counter = document_frequency_selection(global_counter)
        print("po DF " + str(len(global_counter)))
        bin_header = []
        freq_header = []
        normal_freq_header = []
        bin_features = []
        freq_features = []
        normal_freq_features = []
        if len(global_counter) > 0:
            selected_grams = list(global_counter.keys())
            for name in files:
                with open(name) as f:
                    text = f.read()
                    file_size = os.path.getsize(f.name)
                grams = [text[i:i + 2 * n] for i in range(0, (len(text) - 2 * n + 1), 2)]
                grams_freq = collections.Counter(grams)
                bin_feature = [0] * len(selected_grams)
                freq_feature = [0] * len(selected_grams)
                for i in range(len(selected_grams)):
                    if grams_freq[selected_grams[i]] != 0:
                        bin_feature[i] = 1
                        freq_feature[i] = grams_freq[selected_grams[i]]
                bin_features.append(bin_feature)
                freq_features.append(freq_feature)
            bin_features, bin_selected = variance_treshold_selection(bin_features)
            freq_features, freq_selected = variance_treshold_selection(freq_features)
            bin_header = header_from_selection(selected_grams, bin_selected, "")
            freq_header = header_from_selection(selected_grams, freq_selected, "")
            normal_freq_features = freq_features.copy()
            normal_freq_header = freq_header.copy()
            for i in range(len(freq_features)):
                for j in range(len(freq_features[0])):
                    if normal_freq_features[i][j] != 0:
                        normal_freq_features[i][j] = file_size / normal_freq_features[i][j]
                    else:
                        normal_freq_features[i][j] = file_size
    else:
        bin_header = clear_prefix_from_header(bin_prefix)
        freq_header = clear_prefix_from_header(freq_prefix)
        normal_freq_header = clear_prefix_from_header(normal_freq_prefix)
        bin_features = []
        freq_features = []
        normal_freq_features = []
        files = sorted(glob.glob(path))
        for name in files:
            with open(name) as f:
                text = f.read()
                file_size = os.path.getsize(f.name)
            grams = [text[i:i + 2 * n] for i in range(0, (len(text) - 2 * n + 1), 2)]
            grams_freq = collections.Counter(grams)
            bin_feature = [0] * len(bin_header)
            freq_feature = [0] * len(freq_header)
            normal_freq_feature = [0] * len(normal_freq_header)
            for i in range(len(bin_header)):
                for gram in grams:
                    if bin_header[i] in gram:
                        if grams_freq[gram] != 0:
                            bin_feature[i] = 1
            for i in range(len(freq_header)):
                for gram in grams_freq:
                    if freq_header[i] in gram:
                        freq_feature[i] = grams_freq[gram]
            for i in range(len(normal_freq_header)):
                for gram in grams_freq:
                    if normal_freq_header[i] in gram:
                        if grams_freq[gram] != 0:
                            normal_freq_feature[i] = file_size / grams_freq[gram]
                        else:
                            normal_freq_feature[i] = file_size
            bin_features.append(bin_feature)
            freq_features.append(freq_feature)
            normal_freq_features.append(normal_freq_feature)
    return bin_header, bin_features, freq_header, freq_features, normal_freq_header, normal_freq_features


def selected_extraction():
    files = glob.glob(headers_dir)
    for name in files:
        global selected_file
        selected_file = os.path.basename(name)[:-4]
        sample_extraction()
        ngram_extraction()
        os.mkdir(preprocessing.discrete_dir)
        preprocessing.discretize(preprocessing.features_dir, preprocessing.discrete_dir,
                                 preprocessing.discretize_decimals)
        shutil.rmtree(preprocessing.features_dir[:-1])
        os.renames(preprocessing.discrete_dir, preprocessing.original_path[:-2])
        original_file = classification.selected_dir[:-1] + selected_file + ".csv"
        preprocessing.merge_features_from_dir(preprocessing.original_path, original_file)
        shutil.rmtree(preprocessing.original_path[:-1])
        preprocessing.saved_standardize(name, classification.standard_selected_dir)


def sample_extraction():
    prefix = "export"
    header, features = create_export_features(reports_path, prefix)
    features_to_csv(header, features, prefix)

    prefix = "import-libs"
    header, features = create_import_libs_features(reports_path, prefix)
    features_to_csv(header, features, prefix)

    prefix = "import-funcs"
    header, features = create_import_func_features(reports_path, prefix)
    features_to_csv(header, features, prefix)

    prefix = "metadata"
    header, features = create_metadata_features(reports_path, prefix)
    features_to_csv(header, features, prefix)

    prefix = "overlay"
    header, features = create_overlay_features(reports_path, prefix)
    features_to_csv(header, features, prefix)

    prefix = "sections"
    header, features = create_section_features(reports_path, prefix)
    features_to_csv(header, features, prefix)

    prefix = "resources"
    header, features = create_resource_features(reports_path, prefix)
    features_to_csv(header, features, prefix)

    prefix = "strings"
    header, features = create_string_features(string_path, prefix)
    features_to_csv(header, features, prefix)

    prefix = "histogram"
    header, features = create_byte_entropy_histogram_features(hex_path, prefix, step=512, window=2048)
    features_to_csv(header, features, prefix)

    prefix = "sizes"
    header, features = create_sizes_features(reports_path, hex_path, disassembled_path, prefix)
    features_to_csv(header, features, prefix)

    prefix = "file-entropy"
    header, features = create_entropy_feature(entropy_file, prefix)
    features_to_csv(header, features, prefix)

    prefix = "instructions"
    header, features = create_instruction_features(instructions_path, opcodes_path, registers_path, prefix)
    features_to_csv(header, features, prefix)

    prefix = "disassembled"
    header, features = create_disassembled_features(disassembled_path, prefix)
    features_to_csv(header, features, prefix)


def ngram_extraction():
    for i in range(1, max_ngram + 1):
        bin_prefix = str(i) + "-gram-bin-strings"
        freq_prefix = str(i) + "-gram-freq-strings"
        bin_header, bin_features, freq_header, freq_features = create_n_grams(string_path, i, False, bin_prefix,
                                                                              freq_prefix)
        features_to_csv(bin_header, bin_features, bin_prefix)
        features_to_csv(freq_header, freq_features, freq_prefix)

    for i in range(1, max_ngram + 1):
        bin_prefix = str(i) + "-gram-char-bin-strings"
        freq_prefix = str(i) + "-gram-char-freq-strings"
        bin_header, bin_features, freq_header, freq_features = create_n_grams(string_path, i, True, bin_prefix,
                                                                              freq_prefix)
        features_to_csv(bin_header, bin_features, bin_prefix)
        features_to_csv(freq_header, freq_features, freq_prefix)

    for i in range(1, max_ngram + 1):
        bin_prefix = str(i) + "-gram-instructions"
        freq_prefix = str(i) + "-gram-freq-instructions"
        bin_header, bin_features, freq_header, freq_features = create_n_grams(instructions_path, i, False, bin_prefix,
                                                                              freq_prefix)
        features_to_csv(bin_header, bin_features, bin_prefix)
        features_to_csv(freq_header, freq_features, freq_prefix)

    for i in range(1, max_ngram + 1):
        bin_prefix = str(i) + "-gram-reg"
        freq_prefix = str(i) + "-gram-freq-reg"
        bin_header, bin_features, freq_header, freq_features = create_n_grams(registers_path, i, False, bin_prefix,
                                                                              freq_prefix)
        features_to_csv(bin_header, bin_features, bin_prefix)
        features_to_csv(freq_header, freq_features, freq_prefix)

    for i in range(1, max_ngram + 1):
        bin_prefix = str(i) + "-gram-bin-hex"
        freq_prefix = str(i) + "-gram-freq-hex"
        normal_freq_prefix = str(i) + "-gram-normal-freq-hex"
        bin_header, bin_features, freq_header, freq_features, normal_freq_header, normal_freq_features = \
            create_hex_grams(hex_path, i, bin_prefix, freq_prefix, normal_freq_prefix)
        features_to_csv(bin_header, bin_features, bin_prefix)
        features_to_csv(freq_header, freq_features, freq_prefix)
        features_to_csv(normal_freq_header, normal_freq_features, normal_freq_prefix)

    for i in range(1, max_ngram + 1):
        bin_prefix = str(i) + "-gram-opcode"
        freq_prefix = str(i) + "-gram-freq-opcode"
        normal_freq_prefix = str(i) + "-gram-normal-freq-opcode"
        bin_header, bin_features, freq_header, freq_features, normal_freq_header, normal_freq_features = \
            create_hex_grams(opcodes_path, i, bin_prefix, freq_prefix, normal_freq_prefix)
        features_to_csv(bin_header, bin_features, bin_prefix)
        features_to_csv(freq_header, freq_features, freq_prefix)
        features_to_csv(normal_freq_header, normal_freq_features, normal_freq_prefix)


def main():
    # spusti len raz (ak budem mat dalsi dataset musim z danych priecinkov odstranit subory)
    if first_time:
        divide_disassembled_files(disassembled_path, opcodes_path, registers_path, instructions_path)
    if for_prediction:
        selected_extraction()
    else:
        sample_extraction()
        ngram_extraction()


if __name__ == "__main__":
    main()

# tuto skupinu atributov nepouzivam lebo je prilis velka
# for i in range(1, max_ngram + 1):
#     bin_header, bin_features, freq_header, freq_features = create_n_grams(dis_hex_path, i, False)
#     bin_header = [str(i) + "-gram_hex_opcode"]*len(bin_features[0])
#     freq_header = [str(i) + "-gram_opcode_freq_hex"]*len(freq_features[0])
#     features_to_csv(bin_header, bin_features, str(i) + "-gram_hex_opcode")
#     features_to_csv(freq_header, freq_features, str(i) + "-gram_opcode_freq_hex")
