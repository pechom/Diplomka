import json
import re
import glob
import os
import csv
import collections
import pprint

path_labeling = 'C:/PycharmProjects/Diplomka/reports/*'
path_distribution = 'subory/labels2.csv'
# input = open("subory/3_same_with_packed,obfuscated.txt", "r")


def labeling(input, path):  # vytvori csv subor v ktorom budu mena vzoriek a ich triedy
    classes = input.read().splitlines()
    out = open('subory/labels2.csv', 'w')
    out.write("id,trieda" + '\n')
    files = sorted(glob.glob(path))
    wannacry = {"wannacry", "wannacrypt", "wanna", "wannacryptor"}
    for file in files:
        with open(file) as f:
            data = json.load(f)
        kaspersky = data["scans"]["Kaspersky"]["result"].lower()
        mcafee = data["scans"]["McAfee"]["result"].lower()
        eset = data["scans"]["ESET-NOD32"]["result"].lower()
        bitdefender = data["scans"]["BitDefender"]["result"].lower()
        labels = [kaspersky, mcafee, eset, bitdefender]
        for label in labels:
            label = re.split('[_ :/.!,-]', label)
            for trieda in label:
                if trieda in wannacry:
                    trieda = "wannacry"
                if trieda in classes:
                    out.write(os.path.basename(f.name) + "," + trieda + '\n')
                    break
            else:
                continue  # only executed if the inner loop did NOT break
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


def normal_distribution(path_distribution, path_labeling):  # vyhodi vzorky z tried ktore maju neumerne vela vzoriek
    counter = class_distribution(path_distribution)
    files = sorted(glob.glob(path_labeling))
    with open(path_distribution, mode='r') as csv_file, open('subory/new_labels2.csv', 'w', newline='') as out:
        reader = csv.DictReader(csv_file)
        writer = csv.DictWriter(out, fieldnames=["id", "trieda"])
        writer.writeheader()
        names = []
        for line in reader:
            names.append(line['id'])
            if counter[line['trieda']] > 200:
                counter[line['trieda']] -= 1
                for file in files:
                    if os.path.basename(file) == line['id']:
                        os.remove(file)
            else:
                writer.writerow(line)
    for file in files:
        if os.path.basename(file) not in names:
            os.remove(file)


def rename_classes_to_numbers(path_distribution):
    with open(path_distribution, mode='r') as csv_file, open('subory/new_labels2.csv', 'w', newline='') as out:
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


def only_classes_csv(path_distribution):
    with open(path_distribution, mode='r') as csv_file, open('subory/clear_labels2.csv', 'w', newline='') as out:
        reader = csv.DictReader(csv_file)
        writer = csv.writer(out)
        writer.writerow(["class"])
        for line in reader:
            writer.writerow(line['trieda'])


# labeling(input, path_labeling)
# class_distribution(path_distribution)
# normal_distribution(path_distribution, path_labeling)
# class_distribution(path_distribution)
# rename_classes_to_numbers(path_distribution)
# only_classes_csv(path_distribution)
