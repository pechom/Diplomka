import json
import re
import glob
import os
import csv
import collections
import pprint

path_labeling = 'C:/PycharmProjects/Diplomka/reports/*'
path_distribution = 'subory/labels3.csv'
empty_path = 'subory/empty'
input_path = 'subory/same.txt'


def labeling(input_path, path):  # vytvori csv subor v ktorom budu mena vzoriek a ich triedy
    input_classes = open(input_path, "r")
    classes = input_classes.read().splitlines()
    out = open('subory/labels3.csv', 'w')
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
    with open(path_distribution, mode='r') as csv_file, open('subory/new_labels.csv', 'w', newline='') as out:
        reader = csv.DictReader(csv_file)
        writer = csv.DictWriter(out, fieldnames=["id", "trieda"])
        writer.writeheader()
        names = []
        for line in reader:
            names.append(line['id'])
            if counter[line['trieda']] > 120:
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


# labeling(input_path, path_labeling)
# class_distribution(path_distribution)
# normal_distribution(path_distribution, path_labeling)
# class_distribution(path_distribution)
# clear_empty(empty_path, path_distribution)
# rename_classes_to_numbers(path_distribution)
# only_classes_csv(path_distribution)
