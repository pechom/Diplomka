import numpy as np
import glob
import os
import csv
import collections
import re
import matplotlib.pyplot as plt
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sn

best_features_path = 'features/selection/*'
results_path = 'results_third_dataset/'
intersections_file = results_path + 'intersections.txt'
best_groups_output_file = results_path + 'groups.txt'
simple_file = 'features/simple.csv'
very_simple_file = 'features/very_simple.csv'
original_file = 'features/original.csv'
selected_results = results_path + 'selected/*'
compact_selected_results = results_path + 'compact_selected/'
groups_result_path = 'results/*'
difference_file = 'results/3vsall.txt'
labels_path = 'subory/cluster_labels.csv'
predictions_file = results_path + 'predictions_selected.csv'
class_number_file = 'subory/class_number.txt'
prediction_accuracy_file = 'results/prediction_accuracy'


def intersections(input_path, output_path):
    # pocet atributov v ktorych sa prelinaju.
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
    # simple_features = set(np.loadtxt(simple_file, delimiter=',', max_rows=1, dtype="str"))
    # very_simple_features = set(np.loadtxt(very_simple_file, delimiter=',', max_rows=1, dtype="str"))
    files = sorted(glob.glob(input_dir))
    with open(output_file, 'w') as out:
        for file in files:
            groups = collections.Counter()
            # simple = 0
            # very_simple = 0
            header = np.loadtxt(file, delimiter=',', max_rows=1, dtype="str")
            for i in range(len(header)):
                for feature in features:
                    if header[i] == feature or header[i] == ('"' + feature + '"'):
                        groups[feature] += 1
                        # if feature in simple_features:
                        #     simple += 1
                        #     if feature in very_simple_features:
                        #         very_simple += 1
            out.write(os.path.basename(file)[:-4] + '\n')
            out.write("pocet vybranych: " + str(len(header)) + '\n')
            out.write(str(groups.most_common()) + '\n')
            # out.write("simple: " + str(simple) + '\n')
            # out.write("very simple: " + str(very_simple) + '\n')
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
                            results.append(float(splitted[2]))
                            next_line = "  "
                            while next_line != '\n':
                                next_line = next(result_file)  # su tam dve prazdne riadky
                            next(result_file)
                            if next(result_file) == "------------------------------------------------------------\n":
                                break
                        output_file.write('{:.5}'.format(str(min(results) * 100)) + "\t "
                                          + '{:.5}'.format(str(max(results) * 100)) + '\n')
                        next(result_file)
                        next(result_file)
                        results = []
                except StopIteration:
                    continue


def feature_intersections(input_path, output):
    # prienik medzi selektovanymi skupinami v najlepsich vysledkoch (vstup ma v kazdom riadku vysledok jednej selekcie)
    # ak je viac najlepsich vysledkov dam vsetky atributy do jedneho riadku (jednej mnoziny)
    # atrivuty z najlepsich vysledkov som vybral z best_groups vysledkov rucne
    files = sorted(glob.glob(input_path))
    intersections = {}
    for file in files:
        with open(file, "r") as selected_file:
            sets = []
            for count, line in enumerate(selected_file):
                sets.append(set(re.findall(r'\'(.+?)\'', line)))
        intersection = set.intersection(*sets)
        intersections[os.path.basename(file[:-4])] = intersection
        if output:
            print(os.path.basename(file[:-4]))
            print(sorted(intersection))
    if not output:
        return intersections


def feature_intersections_difference(input_path, first, second):
    # rozdiel medzi dvomi skupinami atributov, pouzivam pre rozdiel medzi prienikmi pre stromove a svm vysledky
    intersections = feature_intersections(input_path, False)
    difference = set.difference(intersections[first], intersections[second])
    print(sorted(difference))


def feature_difference(input_path):
    # pouzivam na najdenie rozdielu pre atributy v tretom datasete a ostatnych
    with open(input_path, "r") as file:
        sets = []
        for count, line in enumerate(file):
            sets.append(set(re.findall(r'\'(.+?)\'', line)))
    if len(sets) > 1:
        print(sorted(set.difference(sets[0], sets[1])))


def results_graphs():
    matrix = [[[99.66, 99.67, 99.67], [98.9, 99.45, 94.95],
               [math.log2(79), math.log2(500), math.log2(1000)], [math.log2(766), math.log2(5567), math.log2(486136)],
               [98.9, 99.67, 99.67], [96.5, 98.46, 98.36],
               [math.log2(79), math.log2(500), math.log2(1000)], [math.log2(766), math.log2(5567), math.log2(486136)]],
              [[99.53, 99.69, 99.69], [98.53, 99.15, 94.98],
               [math.log2(48), math.log2(500), math.log2(1000)], [math.log2(518), math.log2(4781), math.log2(536109)],
               [98.53, 99.69, 99.92], [95.13, 97.68, 98.14],
               [math.log2(48), math.log2(500), math.log2(1000)], [math.log2(518), math.log2(4781), math.log2(536109)]],
              [[99.26, 99.27, 99.39], [97.68, 98.05, 89.87],
               [math.log2(62), math.log2(500), math.log2(1000)], [math.log2(622), math.log2(6070), math.log2(478831)],
               [98.66, 99.87, 99.63], [93.91, 97.08, 97.33],
               [math.log2(62), math.log2(500), math.log2(1000)], [math.log2(622), math.log2(6070), math.log2(478831)]]]
    for i in range(len(matrix)):
        plt.plot(matrix[i][2], matrix[i][0], marker='o', color='blue', alpha=1, label="stromy selektované")
        plt.plot(matrix[i][6], matrix[i][4], marker='o', color='red', alpha=1, label="SVM selektované")
        plt.plot(matrix[i][3], matrix[i][1], marker='o', color='darkblue', alpha=1, label="stromy pred selekciou")
        plt.plot(matrix[i][7], matrix[i][5], marker='o', color='darkred', alpha=1, label="SVM pred selekciou")
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, y1, 100))
        plt.gca().set_yticklabels(['{:.1f}%'.format(x) for x in plt.gca().get_yticks()])
        plt.xlabel('počet atribútov (log2)')
        plt.ylabel('presnosť')
        plt.title('výsledky')
        plt.legend()
        plt.show()


def predictions_results():
    # v results urobit heat map a accuracy pre predikcie (cez predikovane vysledky a realne labels
    labels = np.loadtxt(labels_path, delimiter=',', skiprows=1, dtype=np.uint8)
    names = []
    with open(class_number_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for line in reader:
            names.append(line[0])
    with open(predictions_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        with open(prediction_accuracy_file, "w") as output:
            writer = csv.writer(output, delimiter=',')
            while 1 == 1:
                selector = next(reader, None)
                lgbm = next(reader, None)
                xgboost = next(reader, None)
                svc = next(reader, None)
                if selector is not None or lgbm is not None or xgboost is not None or svc is not None:
                    lgbm = np.cast[int](lgbm)
                    xgboost = np.cast[int](xgboost)
                    svc = np.cast[int](svc)
                    writer.writerow(selector)
                    writer.writerow(["lgbm", "xgboost", "svc"])
                    writer.writerow(
                        [accuracy_score(labels, lgbm), accuracy_score(labels, xgboost), accuracy_score(labels, svc)])
                    mat = confusion_matrix(labels, lgbm)  # triedy su zoradene ciselne
                    sn.heatmap(mat, annot=True, annot_kws={"size": 18}, xticklabels=names, yticklabels=names)
                    # plt.show()
                    plt.savefig('subory/' + selector + "lgbm" + '.png')
                    mat = confusion_matrix(labels, xgboost)
                    sn.heatmap(mat, annot=True, annot_kws={"size": 18}, xticklabels=names, yticklabels=names)
                    plt.savefig('subory/' + selector + "xgboost" + '.png')
                    mat = confusion_matrix(labels, svc)
                    sn.heatmap(mat, annot=True, annot_kws={"size": 18}, xticklabels=names, yticklabels=names)
                    plt.savefig('subory/' + selector + "svc" + '.png')
                else:
                    break
    return 0


def main():
    best_groups(best_features_path, best_groups_output_file)
    result_processing(selected_results, compact_selected_results)
    feature_intersections(groups_result_path, True)
    feature_intersections_difference(groups_result_path, 'min_stromy', 'min_spolu')
    feature_difference(difference_file)
    predictions_results()


if __name__ == "__main__":
    main()
