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
wannacry_aliases = {"wannacry", "wannacrypt", "wanna", "wannacryptor"}
antivirus_names = ["kaspersky", "mcafee", "eset", "bitdefender"]

processed_path = 'subory/same.txt'
reports_path = 'reports/*'
consensus_labels_file = 'subory/labels.csv'
cluster_labels_file = 'subory/cluster_labels.txt'
empty_path = 'subory/empty'
long_names_file = 'subory/long_names.txt'
dendogram_file = 'subory/dendogram.png'
levenshtein_file = 'subory/levenshtein_matrix3.txt'
output_dir = 'subory/'

min_class_size = 50  # minimalna velkost triedy
min_cluster_size = 50  # minimalna velkost klastra
min_normal_size = 50  # minimalna velkost triedy po normalizacii distribucie
max_normal_size = 100  # maximalna velkost triedy po normalizacii distribucie
consensus_size = 2  # pocet AV pre ktore ma byt trieda spolocna
name_min_length = 4  # minimalna dlzka mena malweru
labeling_type = "clustering"


def preprocess(path, antivirus_names):
    reports = sorted(glob.glob(path))
    antiviruses = []
    longnames = []
    for name in reports:
        with open(name) as f:
            data = json.load(f)
        names = []
        for antivirus in antivirus_names:
            names.append(data["scans"][antivirus]["result"])
        long_name = ""
        for i in range(len(names)):
            long_name = long_name + names[i] + "#"
        longnames.append(long_name[:-1].lower())

        for i in range(len(names)):
            if len(antiviruses) == len(names):
                antiviruses[i] = antiviruses[i] + names[i] + ","
            else:
                antiviruses = names.copy()
                break
    with open(long_names_file, "w") as file:
        for name in longnames:
            file.write(name + '\n')
    return antiviruses


def classes(antiviruses, class_min):  # vrati mena tried ktore su v jednotlivych antivirusoch dost caste
    final_classes = set({})
    for i in range(len(antiviruses)):
        antivirus = antiviruses[i][:-1]
        antivirus = re.split('[_ :/.!,-]', antivirus)
        longNames = []
        out = open('subory/' + antivirus_names[i] + '.txt', 'w')
        for name in antivirus:
            if len(name) >= name_min_length:
                name = name.lower()
                if name not in generic:
                    if name in wannacry_aliases:
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


def labeling(input_path, path, same, antivirus_names):  # vytvori csv subor v ktorom budu mena vzoriek a ich triedy
    input_classes = open(input_path, "r")
    classes = input_classes.read().splitlines()
    out = open(consensus_labels_file, 'w')
    out.write("id,trieda" + '\n')
    files = sorted(glob.glob(path))
    for file in files:  # pre kazdy report (vzorku) zistujem konsenzus
        with open(file) as f:
            data = json.load(f)
        labels = []
        for name in antivirus_names:
            labels.append(data["scans"][name]["result"].lower())
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
    with open(path_distribution, mode='r') as csv_file, open(output_dir + 'new_labels.csv', 'w', newline='') as out:
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
    os.rename(output_dir + 'new_labels.csv', path_distribution)


def rename_classes_to_numbers(path_distribution):
    with open(path_distribution, mode='r') as csv_file, open(output_dir + 'new_labels.csv', 'w', newline='') as out:
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
    os.rename(output_dir + 'new_labels.csv', path_distribution)


def only_classes_csv(path_distribution):
    with open(path_distribution, mode='r') as csv_file, open(output_dir + 'classes_labels.csv', 'w', newline='') as out:
        reader = csv.DictReader(csv_file)
        writer = csv.writer(out)
        writer.writerow(["class"])
        for line in reader:
            writer.writerow(line['trieda'])
    os.remove(path_distribution)
    os.rename(output_dir + 'classes_labels.csv', path_distribution)


def clear_empty(empty_file, labels_file):  # odstrani polozky pre ktore sa nepodarilo ziskat hex. subor.
    # empty subor mam z remnuxu
    with open(empty_file, mode='r') as empty:
        to_delete = empty.readlines()
    with open(labels_file) as labels:
        to_clean = labels.readlines()
    with open(output_dir + 'clear_labels.csv', mode='w') as out:
        for label in to_clean:
            is_empty = 0
            for empty_label in to_delete:
                if label.startswith(empty_label[:-1]):  # posledny je koniec riadku
                    is_empty = 1
            if not is_empty:
                out.write(label)
    os.remove(labels_file)
    os.rename(output_dir + 'clear_labels.csv', labels_file)


def levenshtein_matrix(input_file):
    with open(input_file, "r") as names_file:
        input_list = names_file.readlines()
    matrix = np.zeros([len(input_list), len(input_list)], dtype=np.uint32)
    with open(levenshtein_file, "w", newline='') as output:
        for i in range(len(input_list)):
            for j in range(i + 1, len(input_list)):
                # nad aj pod diagonalou su rovnake vzdialenosti, diagonala je nulovu
                matrix[i, j] = matrix[j, i] = Levenshtein.distance(input_list[i], input_list[j])
        writer = csv.writer(output, delimiter=',')
        writer.writerows(matrix)


def clustering(min_cluster_size):
    distance_matrix = np.loadtxt(levenshtein_file, delimiter=',', dtype=np.float64)
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                gen_min_span_tree=False, leaf_size=40, metric='precomputed',
                                min_cluster_size=min_cluster_size, min_samples=1, p=None)
    clusterer.fit(distance_matrix)
    c = collections.Counter(clusterer.labels_)
    pprint.pprint(c.most_common(len(c)))
    print(clusterer.labels_.max())
    with open(cluster_labels_file, "w", newline='') as output:
        output.write("class" + '\n')
        for label in clusterer.labels_:
            output.write(str(label) + '\n')
    clusterer.condensed_tree_.plot(select_clusters=True)
    # plt.show()
    plt.savefig(dendogram_file)


def main():
    # preprocessing
    antiviruses = preprocess(reports_path, antivirus_names)
    classes(antiviruses, min_class_size)

    if labeling_type == "consensus":
        # consensus labeling with distribution normalization
        labeling(processed_path, reports_path, consensus_size, antivirus_names)
        class_distribution(consensus_labels_file)
        normal_distribution(consensus_labels_file, reports_path, max_normal_size, min_normal_size)
        class_distribution(consensus_labels_file)
        clear_empty(empty_path, consensus_labels_file)
        rename_classes_to_numbers(consensus_labels_file)
        only_classes_csv(consensus_labels_file)
    else:
        if labeling_type == "clustering":
            # cluster labeling
            levenshtein_matrix(long_names_file)
            clustering(min_cluster_size)
        else:
            print("wrong labeling type")


if __name__ == "__main__":
    main()
