import json
import re
import glob
import os
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
antivirus_names = ["Kaspersky", "McAfee", "ESET-NOD32", "BitDefender"]

processed_path = 'subory/same.txt'
reports_path = 'reports2/*'
consensus_labels_file = 'subory/labels.csv'
cluster_labels_file = 'subory/cluster_labels.csv'
empty_path = 'subory/empty'
long_names_file = 'subory/long_names.txt'
dendogram_file = 'subory/dendogram.png'
levenshtein_file = 'subory/levenshtein_matrix.txt'
output_dir = 'subory/'
class_number_file = 'subory/class_number.txt'

min_class_size = 50  # minimalna velkost triedy
min_cluster_size = 50  # minimalna velkost klastra
min_normal_size = 50  # minimalna velkost triedy po normalizacii distribucie
max_normal_size = 100  # maximalna velkost triedy po normalizacii distribucie
consensus_size = 2  # pocet AV pre ktore ma byt trieda spolocna
name_min_length = 4  # minimalna dlzka mena malweru
labeling_type = "consensus"
for_prediction = True  # ci robim predikciu


def preprocess(path, collect_names):
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
        if collect_names:
            for i in range(len(names)):
                if len(antiviruses) == len(names):
                    antiviruses[i] = antiviruses[i] + names[i] + ","
                else:
                    antiviruses = names.copy()
                    break
    with open(long_names_file, "w") as file:
        for name in longnames:
            file.write(name + '\n')
    if collect_names:
        return antiviruses


def classes(class_min):  # vrati mena tried ktore su v jednotlivych antivirusoch dost caste
    antiviruses = preprocess(reports_path, True)
    final_classes = set({})
    for i in range(len(antiviruses)):
        antivirus = antiviruses[i][:-1]
        antivirus = re.split('[_ :/.!,-]', antivirus)
        long_names = []
        out = open('subory/' + antivirus_names[i] + '.txt', 'w')
        for name in antivirus:
            if len(name) >= name_min_length:
                name = name.lower()
                if name not in generic:
                    if name in wannacry_aliases:
                        name = "wannacry"
                    long_names.append(name)
        c = collections.Counter(long_names)
        # pprint.pprint(c.most_common(len(long_names)))
        for name, count in c.items():
            if count >= class_min:
                final_classes.add(name)
                out.write(name + '\n')
        out.close()
    with open(processed_path, 'w') as out:
        for name in final_classes:
            out.write(name + '\n')


def labeling(input_path, path, same):
    # vytvori csv subor v ktorom budu mena vzoriek a ich triedy pre vzorky ktore dosiahli konsenzus
    input_classes = open(input_path, "r")
    classes = input_classes.read().splitlines()
    out = open(consensus_labels_file, 'w')
    out.write("id,class" + '\n')
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


def class_distribution(path):  # zisti kolko vzoriek je v jednotlivych triedach
    labels = []
    with open(path, mode='r') as csv_file:
        reader = csv.DictReader(csv_file)
        for line in reader:
            labels.append(line['class'])
        c = collections.Counter(labels)
        pprint.pprint(c.most_common(len(labels)))
    csv_file.close()
    return c


def normal_distribution(path_labels, path_reports, max_size, min_size):
    # vyhodi reporty z tried ktore maju neumerne vela alebo malo vzoriek. Odstrani outliere (nedosiahli konsenzus)
    # potom je potrebne odstranit aj samotne vzorky odstranenych reportov - robim vo virtualke
    counter = class_distribution(path_labels)
    files = sorted(glob.glob(path_reports))
    with open(path_labels, mode='r') as csv_file, open(output_dir + 'new_labels.csv', 'w', newline='') as out:
        reader = csv.DictReader(csv_file)
        writer = csv.DictWriter(out, fieldnames=["id", "class"])
        writer.writeheader()
        names = []
        for line in reader:
            names.append(line['id'])
            if (counter[line['class']] > max_size or (0 < counter[line['class']] < min_size)) or line['class'] == "-1":
                counter[line['class']] -= 1
                for file in files:
                    if os.path.basename(file) == line['id']:
                        os.remove(file)
                        break
            else:
                writer.writerow(line)
    for file in files:
        if os.path.basename(file) not in names:  # tieto vzorky nedosiahli konsenzus
            os.remove(file)
    os.remove(path_labels)
    os.rename(output_dir + 'new_labels.csv', path_labels)


def delete_unwanted_classes(path_labels, path_reports, class_number_file):
    # odstrani z predikovaneho datasetu triedy ktore neboli v trenovacom a outliere
    files = sorted(glob.glob(path_reports))
    names = []
    with open(path_labels, mode='r') as csv_file, open(class_number_file, 'r') as class_file, open(
            output_dir + 'new_labels.csv', 'w', newline='') as out:
        reader = csv.DictReader(csv_file)
        class_reader = csv.DictReader(class_file)
        classes = set()
        for line in class_reader:
            classes.add(line['class'])
        writer = csv.DictWriter(out, fieldnames=["id", "class"])
        writer.writeheader()
        for line in reader:
            names.append(line['id'])
            if line['class'] not in classes:  # trieda ktora nebola v trenovacom datasete
                for file in files:
                    if os.path.basename(file) == line['id']:
                        os.remove(file)
                        break
            else:
                writer.writerow(line)
    for file in files:
        if os.path.basename(file) not in names:  # tieto vzorky nedosiahli konsenzus
            os.remove(file)
    os.remove(path_labels)
    os.rename(output_dir + 'new_labels.csv', path_labels)


def rename_classes_to_numbers(path_labels):
    with open(path_labels, mode='r') as csv_file, open(output_dir + 'new_labels.csv', 'w', newline='') as out:
        reader = csv.DictReader(csv_file)
        writer = csv.DictWriter(out, fieldnames=["id", "class"])
        writer.writeheader()
        classes = set()
        for line in reader:
            classes.add(line['class'])
        classes = list(classes)
        # print(classes)
        csv_file.seek(0)
        next(reader)
        with open(class_number_file, 'w', newline='') as out_file:
            # triedy ktore mozu mat predikovane vzorky
            out_file.write("class,number" + '\n')
            for trieda in classes:
                out_file.write(trieda + ',' + str(classes.index(trieda)) + '\n')
        for line in reader:
            line['class'] = classes.index(line['class'])
            writer.writerow(line)
    os.remove(path_labels)
    os.rename(output_dir + 'new_labels.csv', path_labels)


def keep_classes_numbers(path_labels, path_classes):
    # prepise triedy na cisla v predikovanom datasete so zachovanim cisel z trenovacieho
    with open(path_labels, mode='r') as csv_file, open(path_classes, 'r') as class_file, open(
            output_dir + 'new_labels.csv', 'w', newline='') as out:
        reader = csv.DictReader(csv_file)
        class_reader = csv.DictReader(class_file)
        class_numbers = {line['class']: line['number'] for line in class_reader}
        writer = csv.DictWriter(out, fieldnames=["id", "class"])
        writer.writeheader()
        for line in reader:
            line['class'] = class_numbers[line['class']]
            writer.writerow(line)
    os.remove(path_labels)
    os.rename(output_dir + 'new_labels.csv', path_labels)


def only_classes_csv(path_distribution):
    with open(path_distribution, mode='r') as csv_file, open(output_dir + 'classes_labels.csv', 'w', newline='') as out:
        reader = csv.DictReader(csv_file)
        writer = csv.writer(out)
        writer.writerow(["class"])
        for line in reader:
            writer.writerow(line['class'])
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


def clustering(reports_path, min_cluster_size):
    distance_matrix = np.loadtxt(levenshtein_file, delimiter=',', dtype=np.float64)
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                gen_min_span_tree=False, leaf_size=40, metric='precomputed',
                                min_cluster_size=min_cluster_size, min_samples=1, p=None)
    clusterer.fit(distance_matrix)
    c = collections.Counter(clusterer.labels_)
    pprint.pprint(c.most_common(len(c)))
    print(clusterer.labels_.max())
    files = sorted(glob.glob(reports_path))
    counter = 0
    with open(cluster_labels_file, "w", newline='') as output:
        output.write("id,class" + '\n')
        for label in clusterer.labels_:
            output.write(os.path.basename(files[counter]) + "," + str(label) + '\n')
            counter += 1
    clusterer.condensed_tree_.plot(select_clusters=True)
    # plt.show()
    plt.savefig(dendogram_file)


def main():
    if labeling_type == "consensus":
        classes(min_class_size)
        # consensus labeling with distribution normalization
        labeling(processed_path, reports_path, consensus_size)
        class_distribution(consensus_labels_file)
        if not for_prediction:
            normal_distribution(consensus_labels_file, reports_path, max_normal_size, min_normal_size)
        else:
            delete_unwanted_classes(consensus_labels_file, reports_path, class_number_file)
        class_distribution(consensus_labels_file)
        clear_empty(empty_path, consensus_labels_file)
        if not for_prediction:
            rename_classes_to_numbers(consensus_labels_file)
        else:
            keep_classes_numbers(consensus_labels_file, class_number_file)
        only_classes_csv(consensus_labels_file)
    else:
        if labeling_type == "clustering":
            preprocess(reports_path, False)
            # cluster labeling with distribution normalization
            levenshtein_matrix(long_names_file)
            clustering(reports_path, min_cluster_size)
            normal_distribution(cluster_labels_file, reports_path, max_normal_size, min_normal_size)
            class_distribution(cluster_labels_file)
            clear_empty(empty_path, cluster_labels_file)
            only_classes_csv(cluster_labels_file)
        else:
            print("wrong labeling type")


if __name__ == "__main__":
    main()
