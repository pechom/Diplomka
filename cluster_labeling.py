import Levenshtein
import numpy as np
import hdbscan
import csv
import collections
import pprint
import matplotlib.pyplot as plt

input = open("stare subory/long_names.txt", "r")

list1 = input.readlines()
input.close()


def same_groups(input_list):
    already_checked_groups = []
    counter = 0
    with open("stare subory/levenshtein.txt", "w") as output:
        for i in range(len(input_list)):
            if i != 0 and counter > 49:
                already_checked_groups.append(input_list[i-1])
                output.write(input_list[i-1])
                output.write(str(counter) + '\n')
            counter = 1
            for j in range(i+1, len(input_list)):
                if input_list[j] not in already_checked_groups:
                    if input_list[i] == input_list[j]:
                        counter += 1
    print(len(already_checked_groups))


def levenshtein_matrix(input_list):
    matrix = np.zeros([len(input_list), len(input_list)], dtype=np.uint32)
    with open("stare subory/levenshtein_matrix.txt", "w", newline='') as output:
        for i in range(len(input_list)):
            for j in range(len(input_list)):
                matrix[i, j] = Levenshtein.distance(input_list[i], input_list[j])
        writer = csv.writer(output, delimiter=',')
        writer.writerows(matrix)


def clustering():
    distance_matrix = np.loadtxt("stare subory/levenshtein_matrix.txt", delimiter=',', dtype=np.float64)
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                gen_min_span_tree=False, leaf_size=40, metric='precomputed',
                                min_cluster_size=86, min_samples=1, p=None)
    clusterer.fit(distance_matrix)
    c = collections.Counter(clusterer.labels_)
    pprint.pprint(c.most_common(len(c)))
    print(clusterer.labels_.max())
    # with open("stare subory/cluster_labels.txt", "w", newline='') as output:
    #     output.write("class" + '\n')
    #     for label in clusterer.labels_:
    #         output.write(str(label)+'\n')
    # clusterer.condensed_tree_.plot(select_clusters=True)
    # plt.show()
    # plt.savefig('stare subory/dendogram.png')


# levenshtein_matrix(list1)
# same_groups(list1)
clustering()