import json
import re
import glob
import os
import csv
import collections
import pprint
import numpy as np
import hdbscan
import Levenshtein
import csv
import matplotlib.pyplot as plt

generic = {"win32", "variant", "spyware", "trojan", "worm", "virus", "heur", "trojandownloader", "generic",
           "malware", "ransom", "ransomware", "trojandropper", "autorun", "agent", "adware", "downloader", "infector",
           "backdoor", "hacktool", "bitcoinminer", "win64", "potentially", "toolbar", "unwanted", "small", "tiny",
           "blocker", "proxy", "email", "injector", "inject", "softwarebundler", "virtool", "ddos", "exploit",
           "filecoder", "dropper", "sitehijack", "cryptolocker", "application", "deepscan", "keylogger", "obfuscated",
           "packed", "encrypted", "packer", "obfuscator", "encryptor", "malpack", "startpage", "servstart",
           "infostealer", "crypt", "rootkit", "passwordstealer", "optional", "genpack"}
wannacry = {"wannacry", "wannacrypt", "wanna", "wannacryptor"}
antivirus_names = ["kaspersky", "mcafee", "eset", "bitdefender"]

bitdefender_file = 'subory/bitdefender.txt'
kaspersky_file = 'subory/kaspersky.txt'
mcafee_file = 'subory/mcafee.txt'
eset_file = 'subory/eset.txt'
processed_path = 'subory/same.txt'
reports_path = 'C:/PycharmProjects/Diplomka/reports/*'
path_labeling = 'C:/PycharmProjects/Diplomka/reports/*'
path_distribution = 'subory/labels3.csv'
empty_path = 'subory/empty'
long_names_file = 'subory/long_names3.txt'


def preprocess(path):
    reports = sorted(glob.glob(path))
    names_kaspersky = names_mcafee = names_eset = names_bitdefender = ""
    longnames = []
    for name in reports:
        with open(name) as f:
            data = json.load(f)
        kaspersky = data["scans"]["Kaspersky"]["result"]
        mcafee = data["scans"]["McAfee"]["result"]
        eset = data["scans"]["ESET-NOD32"]["result"]
        bitdefender = data["scans"]["BitDefender"]["result"]
        # names = names + microsoft + "," + kaspersky + mcafee + ","
        long_name = kaspersky+'#'+mcafee+'#'+eset+'#'+bitdefender
        longnames.append(long_name.lower())
        names_kaspersky = names_kaspersky + kaspersky + ","
        names_mcafee = names_mcafee + mcafee + ","
        names_eset = names_eset + eset + ","
        names_bitdefender = names_bitdefender + bitdefender + ","
    with open("subory/long_names3.txt", "w") as file:
        for name in longnames:
            file.write(name+'\n')
    antiviruses = [names_kaspersky, names_mcafee, names_eset, names_bitdefender]
    return antiviruses


def classes(antiviruses, class_min):  # vrati mena tried ktore su v jednotlivych antivirusoch dost caste
    final_classes = set({})
    for i in range(len(antiviruses)):
        antivirus = antiviruses[i]
        antivirus = antivirus[:-1]
        antivirus = re.split('[_ :/.!,-]', antivirus)
        longNames = []
        out = open('subory/' + antivirus_names[i] + '.txt', 'w')
        for name in antivirus:
            if len(name) > 3:
                name = name.lower()
                if name not in generic:
                    if name in wannacry:
                        name = "wannacry"
                    longNames.append(name)
        c = collections.Counter(longNames)
        # pprint.pprint(c.most_common(len(longNames)))
        for name, count in c.items():
            if count >= class_min:
                final_classes.add(name)
                out.write(name + '\n')
        out.close()
    with open(processed_path, 'w') as out:
        for name in final_classes:
            out.write(name + '\n')


def labeling(input_path, path, same):  # vytvori csv subor v ktorom budu mena vzoriek a ich triedy
    input_classes = open(input_path, "r")
    classes = input_classes.read().splitlines()
    out = open('subory/labels3.csv', 'w')
    out.write("id,trieda" + '\n')
    files = sorted(glob.glob(path))
    for file in files:
        with open(file) as f:
            data = json.load(f)
        kaspersky = data["scans"]["Kaspersky"]["result"].lower()
        mcafee = data["scans"]["McAfee"]["result"].lower()
        eset = data["scans"]["ESET-NOD32"]["result"].lower()
        bitdefender = data["scans"]["BitDefender"]["result"].lower()
        labels = [kaspersky, mcafee, eset, bitdefender]
        class_counter = collections.Counter(classes)
        for label in labels:
            for trieda in classes:
                if trieda in label:
                    class_counter[trieda] += 1
                if class_counter[trieda] == same + 1:
                    out.write(os.path.basename(f.name) + "," + trieda + '\n')
                    break
            else:
                continue
            break
    out.close()
    f.close()


def class_distribution(path):  # zisti kolko vzoriek je v jednotlivych triedach
    labels = []
    with open(path, mode='r') as csv_file:
        reader = csv.DictReader(csv_file)
        for line in reader:
            labels.append(line['trieda'])
        c = collections.Counter(labels)
        pprint.pprint(c.most_common(len(labels)))
    csv_file.close()
    return c


def normal_distribution(path_distribution, path_labeling, max, min):
    # vyhodi vzorky z tried ktore maju neumerne vela alebo malo vzoriek
    counter = class_distribution(path_distribution)
    files = sorted(glob.glob(path_labeling))
    with open(path_distribution, mode='r') as csv_file, open('subory/new_labels.csv', 'w', newline='') as out:
        reader = csv.DictReader(csv_file)
        writer = csv.DictWriter(out, fieldnames=["id", "trieda"])
        writer.writeheader()
        names = []
        for line in reader:
            names.append(line['id'])
            if counter[line['trieda']] > max or (0 <= counter[line['trieda']] < min):
                counter[line['trieda']] -= 1
                for file in files:
                    if os.path.basename(file) == line['id']:
                        os.remove(file)
            else:
                writer.writerow(line)
    for file in files:
        if os.path.basename(file) not in names:
            os.remove(file)
    os.remove(path_distribution)
    os.rename('subory/new_labels.csv', path_distribution)


def rename_classes_to_numbers(path_distribution):
    with open(path_distribution, mode='r') as csv_file, open('subory/new_labels.csv', 'w', newline='') as out:
        reader = csv.DictReader(csv_file)
        writer = csv.DictWriter(out, fieldnames=["id", "trieda"])
        writer.writeheader()
        classes = set()
        for line in reader:
            classes.add(line['trieda'])
        classes = list(classes)
        print(classes)
        csv_file.seek(0)
        next(reader)
        for line in reader:
            line['trieda'] = classes.index(line['trieda'])
            writer.writerow(line)
    os.remove(path_distribution)
    os.rename('subory/new_labels.csv', path_distribution)


def only_classes_csv(path_distribution):
    with open(path_distribution, mode='r') as csv_file, open('subory/classes_labels.csv', 'w', newline='') as out:
        reader = csv.DictReader(csv_file)
        writer = csv.writer(out)
        writer.writerow(["class"])
        for line in reader:
            writer.writerow(line['trieda'])
    os.remove(path_distribution)
    os.rename('subory/classes_labels.csv', path_distribution)


def clear_empty(empty_file, labels_file):  # odstrani polozky pre ktore sa nepodarilo ziskat hex. subor.
    # empty subor mam z remnuxu
    with open(empty_file, mode='r') as empty:
        to_delete = empty.readlines()
    with open(labels_file) as labels:
        to_clean = labels.readlines()
    with open('subory/clear_labels.csv', mode='w') as out:
        for label in to_clean:
            is_empty = 0
            for empty_label in to_delete:
                if label.startswith(empty_label[:-1]):  # posledny je koniec riadku
                    is_empty = 1
            if not is_empty:
                out.write(label)
    os.remove(labels_file)
    os.rename('subory/clear_labels.csv', labels_file)


def levenshtein_matrix(input_file):
    with open(input_file, "r") as names_file:
        input_list = names_file.readlines()
    matrix = np.zeros([len(input_list), len(input_list)], dtype=np.uint32)
    with open("stare subory/levenshtein_matrix3.txt", "w", newline='') as output:
        for i in range(len(input_list)):
            for j in range(len(input_list)):
                matrix[i, j] = Levenshtein.distance(input_list[i], input_list[j])
        writer = csv.writer(output, delimiter=',')
        writer.writerows(matrix)


def clustering(min_cluster_size):
    distance_matrix = np.loadtxt("subory/levenshtein_matrix3.txt", delimiter=',', dtype=np.float64)
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                gen_min_span_tree=False, leaf_size=40, metric='precomputed',
                                min_cluster_size=min_cluster_size, min_samples=1, p=None)
    clusterer.fit(distance_matrix)
    c = collections.Counter(clusterer.labels_)
    pprint.pprint(c.most_common(len(c)))
    print(clusterer.labels_.max())
    with open("subory/cluster_labels.txt", "w", newline='') as output:
        output.write("class" + '\n')
        for label in clusterer.labels_:
            output.write(str(label)+'\n')
    clusterer.condensed_tree_.plot(select_clusters=True)
    # plt.show()
    plt.savefig('stare subory/dendogram3.png')


# preprocessing
# antiviruses = preprocess(reports_path)
# classes(antiviruses, 50)

# consensus labeling
# same = 2  # pocet AV pre ktore ma byt trieda spolocna
# labeling(input_path, path_labeling, same)
# class_distribution(path_distribution)
# normal_distribution(path_distribution, path_labeling, 100, 50)
# class_distribution(path_distribution)
# clear_empty(empty_path, path_distribution)
# rename_classes_to_numbers(path_distribution)
# only_classes_csv(path_distribution)

# cluster labeling
# levenshtein_matrix(long_names_file)
# clustering(50)
