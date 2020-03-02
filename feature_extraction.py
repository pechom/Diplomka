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

reports_path = 'reports/*'
string_path = 'strings/*'
hex_path = 'hex/*'
disassembled_path = 'disassembled/*'
entropy_file = 'subory/entropy.csv'
dis_hex_path = 'disassembled_divided/hex/*'
registers_path = 'disassembled_divided/registers/*'
opcodes_path = 'disassembled_divided/opcodes/*'

df_min_count = 10  # minimalny pocet vyskytov pre DF pri ktorom odstranim atribut
max_ngram = 2  # maximalne n pre ktore robim n-gram


def document_frequency_selection(counter):
    for name, count in counter.copy().items():
        if count < 10:
            del counter[name]
    return counter


def variance_treshold_selection(data):
    treshold = 0.01
    if len(data[0]) >= 1000:
        treshold = 0.05
    if len(data[0]) >= 10000:
        treshold = 0.1
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
    if not (len(data) == 0):
        with open("features/" + name + ".csv", "w", newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(header)
            writer.writerows(data)
        print("finalna dlzka " + str(len(header)))
    else:
        print("finalna dlzka" + " 0")


def create_import_libs_features(path):
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
                if selected_libs[i] in data["additional_info"]["imports"]:
                    feature[i] = len(data["additional_info"]["imports"][selected_libs[i]])  # pocet funkcii
            feature[-1] = len(data["additional_info"]["imports"])  # pocet kniznic
            features.append(feature)
        features, selected = variance_treshold_selection(features)
        header = header_from_selection(selected_libs, selected, "number_of_DLLs")
    return header, features


def header_from_selection(selected, variance_selected, message):
    header = []
    for i in range(len(selected)):
        if i in variance_selected:
            header.append(selected[i])
    if len(selected) in variance_selected:
        header.append(message)
    return header


def create_import_func_features(path):
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
    header = []
    for i in range(len(selected_funcs)):
        if i in selected:
            header.append(selected_funcs[i])
    return header, features


def create_export_features(path):
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
        header_from_selection(selected_libs, selected, "number_of_DLLs")
    return header, features


def create_metadata_features(path):
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
        # feature[7] = int(data["additional_info"]["exiftool"]["EntryPoint"])
        # feature[6] = data["additional_info"]["pe-timestamp"]
        feature[7] = data["additional_info"]["pe-entry-point"]
        feature[8] = data["additional_info"]["exiftool"]["SubsystemVersion"]
        feature[9] = data["additional_info"]["pe-machine-type"]
        feature[10] = data["size"]
        features.append(feature)
    features, header = delete_not_selected(features, header)
    return header, features


def create_overlay_features(path):
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
    return header, features


def create_section_features(path):
    files = sorted(glob.glob(path))
    features = []
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
    return header, features


def delete_not_selected(features, header):
    features, selected = variance_treshold_selection(features)
    for i in reversed(range(len(header))):
        if i not in selected:
            del header[i]
    return features, header


def create_resource_features(path):
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
                        feature[i] = data["additional_info"]["pe-resource-types"][
                            selected_types[i]]  # pocet resources typu
                        all_resources = 0
                        for resource_type in data["additional_info"]["pe-resource-types"]:
                            all_resources += data["additional_info"]["pe-resource-types"][resource_type]
                feature[-1] = all_resources
            features.append(feature)
        features, selected = variance_treshold_selection(features)
        header_from_selection(selected_types, selected, "number_of_resources")
    return header, features


def create_n_grams(path, n, is_char):
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
        for i in range(len(selected_grams)):
            if i in bin_selected:
                bin_header.append(str(selected_grams[i]))
            if i in freq_selected:
                freq_header.append(str(selected_grams[i]))
    return bin_header, bin_features, freq_header, freq_features


def create_string_features(path):
    files = sorted(glob.glob(path))
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
    for i in range(len(bins) - 1):
        header.append(str(bins[i]) + "-" + str(bins[i + 1]))
    header.extend(["C:\\", "http", "HKEY_", "MZ", "IP", "average_length", "number_of_strings", "file_size", "entropy"])
    for name in files:
        feature = []
        line_lengths = []
        with open(name) as f:
            text = f.readlines()
            for line in text:
                line_lengths.append(len(line))
                max_length = max(max_length, len(line))
            hist = np.histogram(line_lengths, bins=bins)
            feature.extend(hist[0])
            f.seek(0)
            text = f.read()
            low_text = text.lower()
            feature.append(low_text.count("c:\\"))
            feature.append(low_text.count("http"))
            feature.append(low_text.count("hkey_"))
            feature.append(text.count("MZ"))
            feature.append(len(re.findall(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", text)))
            feature.append(np.mean(line_lengths))
            feature.append(len(line_lengths))
            feature.append(os.path.getsize(f.name))
            feature.append(entropy(text))
        features.append(feature)
    features, header = delete_not_selected(features, header)
    return header, features


def create_hex_grams(path, n):
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
    bin_features = []
    freq_features = []
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
        for i in range(len(selected_grams)):
            if i in bin_selected:
                bin_header.append(str(selected_grams[i]))
            if i in freq_selected:
                freq_header.append(str(selected_grams[i]))
    normal_freq_features = freq_features.copy()
    for i in range(len(freq_features)):
        for j in range(len(freq_features[0])):
            if normal_freq_features[i][j] != 0:
                normal_freq_features[i][j] = file_size / normal_freq_features[i][j]
            else:
                normal_freq_features[i][j] = file_size
    return bin_header, bin_features, freq_header, freq_features, freq_header, normal_freq_features


def entropy_bin_counts(block, window):
    # coarse histogram, 16 bytes per bin
    c = np.bincount(block >> 4, minlength=16)  # 16-bin histogram
    p = c.astype(np.float32) / window
    wh = np.where(c)[0]
    h = np.sum(-p[wh] * np.log2(
        p[wh])) * 2  # * x2 b.c. we reduced information by half: 256 bins (8 bits) to 16 bins (4 bits)
    hbin = int(h * 2)  # up to 16 bins (max entropy is 8 bits)
    if hbin == 16:  # handle entropy = 8.0 bits
        hbin = 15
    return hbin, c


def byte_entropy_histogram(bytez, step, window):
    # kod mam z tade https://arxiv.org/pdf/1508.03096.pdf
    output = np.zeros((16, 16), dtype=np.int)
    a = np.frombuffer(bytez, dtype=np.uint8)
    if a.shape[0] < window:
        hbin, c = entropy_bin_counts(a, window)
        output[hbin, :] += c
    else:
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step, :]
        # from the blocks, compute histogram
        for block in blocks:
            hbin, c = entropy_bin_counts(block, window)
            output[hbin, :] += c
    return output.flatten().tolist()


def create_byte_entropy_histogram_features(path, step, window):
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
            feature = byte_entropy_histogram(bytez, step, window)  # krok a okno, vyskusal som viac
            for i in reversed(range(len(feature))):
                if i not in selected_indices:
                    del feature[i]
            features.append(feature)
        features, selected = variance_treshold_selection(features)
        for i in range(len(selected_indices)):
            if i in selected:
                header.append(str(selected_indices[i]))
    return header, features


def create_sizes_features(reports_path, hex_path, disassembled_path):
    files = sorted(glob.glob(reports_path))
    features = []
    header = ["file_size", "hex_size", "disassembled_size", "file/hex_size", "file/disassembled_size", "hex/file_size",
              "hex/disassembled_size", "disassembled/file_size", "disassembled/hex_size"]
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
    return header, features


def create_entropy_feature(file):
    header = ["file_entropy"]
    features = []
    with open(file) as f:
        lines = f.readlines()
    for line in lines:  # subory su rovnako usporiadane ako ked prechadzam priecinok cez glob, testoval som to
        feature = [0] * 1
        feature[0] = line.split()[1]
        features.append(feature)
    return header, features


def divide_disassembled_files(disassembled_path, hex_path, registers_path, opcodes_path):
    files = sorted(glob.glob(disassembled_path))
    for name in files:
        with open(name, errors='replace') as f:
            basename = os.path.basename(f.name)
            text = f.readlines()
        del text[0:7]  # v prvych riadkoch nie su instrukcie
        with open(hex_path[:-1] + basename, 'w') as f, open(opcodes_path[:-1] + basename, 'w') as f2, open(
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


def create_instruction_features(opcodes_path, dis_hex_path, registers_path):
    files = sorted(glob.glob(opcodes_path))
    features = []
    header = ["number_of_allocation", "number_of_jump", "all/allocation",
              "all/jump", "all", "hex", "registers"]
    allocation_instructions = ["dd", "db", "dw", "dup"]
    jump_instructions = ["jmp", "je", "jne", "jg", "jge", "ja", "jae", "jl", "jle", "jb", "jbe", "jo", "jno", "jz",
                         "jnz", "js", "jns", "jcxz", "jecxz", "jrcxz"]
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
    return header, features


def create_disassembled_features(path):
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
                header.append("line_length_" + str(selected_sizes[i]))
        if len(selected_sizes) in selected:
            header.append("average_length")
        if (len(selected_sizes) + 1) in selected:
            header.append("number_of_lines")
    return header, features


def main():
    # spusti len raz, ak budem zase spustat extrakciu atributov toto vynecham !!!
    # (ak budem mat dalsi dataset musim z danych priecinkov odstranit subory)
    divide_disassembled_files(disassembled_path, dis_hex_path, registers_path, opcodes_path)

    # ------------------------------------------

    header, features = create_export_features(reports_path)
    header = ["export"] * len(features[0])
    features_to_csv(header, features, "export")

    header, features = create_import_libs_features(reports_path)
    header = ["import_libs"] * len(features[0])
    features_to_csv(header, features, "import_libs")

    header, features = create_import_func_features(reports_path)
    header = ["import_funcs"] * len(features[0])
    features_to_csv(header, features, "import_funcs")

    header, features = create_metadata_features(reports_path)
    header = ["metadata"] * len(features[0])
    features_to_csv(header, features, "metadata")

    header, features = create_overlay_features(reports_path)
    header = ["overlay"] * len(features[0])
    features_to_csv(header, features, "overlay")

    header, features = create_section_features(reports_path)
    header = ["sections"] * len(features[0])
    features_to_csv(header, features, "sections")

    header, features = create_resource_features(reports_path)
    header = ["resources"] * len(features[0])
    features_to_csv(header, features, "resources")

    header, features = create_string_features(string_path)
    header = ["strings"] * len(features[0])
    features_to_csv(header, features, "strings")

    header, features = create_byte_entropy_histogram_features(hex_path, step=512, window=2048)
    header = ["byte_entropy_histogram"] * len(features[0])
    features_to_csv(header, features, "byte_entropy_histogram")

    header, features = create_sizes_features(reports_path, hex_path, disassembled_path)
    header = ["sizes"] * len(features[0])
    features_to_csv(header, features, "sizes")

    header, features = create_entropy_feature(entropy_file)
    header = ["file_entropy"] * len(features[0])
    features_to_csv(header, features, "file_entropy")

    header, features = create_instruction_features(opcodes_path, dis_hex_path, registers_path)
    header = ["instructions"] * len(features[0])
    features_to_csv(header, features, "instructions")

    header, features = create_disassembled_features(disassembled_path)
    header = ["disassembled"] * len(features[0])
    features_to_csv(header, features, "disassembled")

    # ------------------------------------------

    for i in range(1, max_ngram + 1):
        bin_header, bin_features, freq_header, freq_features = create_n_grams(string_path, i, False)
        bin_header = [str(i) + "-gram_bin_strings"] * len(bin_features[0])
        freq_header = [str(i) + "-gram_freq_strings"] * len(freq_features[0])
        features_to_csv(bin_header, bin_features, str(i) + "-gram_bin_strings")
        features_to_csv(freq_header, freq_features, str(i) + "-gram_freq_strings")

    for i in range(1, max_ngram + 1):
        bin_header, bin_features, freq_header, freq_features, normal_freq_header, normal_freq_features = \
            create_hex_grams(hex_path, i)
        bin_header = [str(i) + "-gram_bin_hex"] * len(bin_features[0])
        freq_header = [str(i) + "-gram_freq_hex"] * len(freq_features[0])
        normal_freq_header = [str(i) + "-gram_normal_freq_hex"] * len(normal_freq_features[0])
        features_to_csv(bin_header, bin_features, str(i) + "-gram_bin_hex")
        features_to_csv(freq_header, freq_features, str(i) + "-gram_freq_hex")
        features_to_csv(normal_freq_header, normal_freq_features, str(i) + "-gram_normal_freq_hex")

    for i in range(1, max_ngram + 1):
        bin_header, bin_features, freq_header, freq_features = create_n_grams(string_path, i, True)
        bin_header = [str(i) + "-gram_char_bin_strings"] * len(bin_features[0])
        freq_header = [str(i) + "-gram_char_freq_strings"] * len(freq_features[0])
        features_to_csv(bin_header, bin_features, str(i) + "-gram_char_bin_strings")
        features_to_csv(freq_header, freq_features, str(i) + "-gram_char_freq_strings")

    for i in range(1, max_ngram + 1):
        bin_header, bin_features, freq_header, freq_features = create_n_grams(opcodes_path, i, False)
        bin_header = [str(i) + "-gram_opcodes"] * len(bin_features[0])
        freq_header = [str(i) + "-gram_freq_opcodes"] * len(freq_features[0])
        features_to_csv(bin_header, bin_features, str(i) + "-gram_opcodes")
        features_to_csv(freq_header, freq_features, str(i) + "-gram_freq_opcodes")

    for i in range(1, max_ngram + 1):
        bin_header, bin_features, freq_header, freq_features = create_n_grams(registers_path, i, False)
        bin_header = [str(i) + "-gram_reg"] * len(bin_features[0])
        freq_header = [str(i) + "-gram_freq_reg"] * len(freq_features[0])
        features_to_csv(bin_header, bin_features, str(i) + "-gram_reg")
        features_to_csv(freq_header, freq_features, str(i) + "-gram_freq_reg")

    for i in range(1, max_ngram + 1):
        bin_header, bin_features, freq_header, freq_features, normal_freq_header, normal_freq_features = \
            create_hex_grams(dis_hex_path, i)
        bin_header = [str(i) + "-gram_dis_hex"] * len(bin_features[0])
        freq_header = [str(i) + "-gram_freq_dis_hex"] * len(freq_features[0])
        normal_freq_header = [str(i) + "-gram_normal_freq_dis_hex"] * len(normal_freq_features[0])
        features_to_csv(bin_header, bin_features, str(i) + "-gram_dis_hex")
        features_to_csv(freq_header, freq_features, str(i) + "-gram_freq_dis_hex")
        features_to_csv(normal_freq_header, normal_freq_features, str(i) + "-gram_normal_freq_dis_hex")


if __name__ == "__main__":
    main()

# tuto skupinu atributov nepouzivam lebo je prilis velka
# for i in range(1, max_ngram + 1):
#     bin_header, bin_features, freq_header, freq_features = create_n_grams(dis_hex_path, i, False)
#     bin_header = [str(i) + "-gram_hex_opcode"]*len(bin_features[0])
#     freq_header = [str(i) + "-gram_opcode_freq_hex"]*len(freq_features[0])
#     features_to_csv(bin_header, bin_features, str(i) + "-gram_hex_opcode")
#     features_to_csv(freq_header, freq_features, str(i) + "-gram_opcode_freq_hex")
