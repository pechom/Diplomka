import itertools
import glob
import os
import csv

path = 'C:/PycharmProjects/Diplomka/skusobny/selection/*'
out_path = 'C:/PycharmProjects/Diplomka/skusobny/intersections.txt'


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
        for k in range(len(headers)):
            if i != k:
                if len(headers[i].intersection(headers[k])) == best:
                    outputs.append(
                        names[i] + " - velkost: " + str(len(headers[i])) + "," + " najlepsi prienik je " + names[
                            k] + ": " + str(len(headers[i].intersection(headers[k]))) + '\n')
    with open(output_path, 'w') as out:
        for output in outputs:
            out.write(output)


intersections(path, out_path)

# with open('C:/PycharmProjects/Diplomka/skusobny/classification/3-gram_bin_strings.csv') as ff:
#     reader = csv.reader(ff, delimiter=',')
#     header = next(reader)
#     for head in header:
#         print(head)
#
