import itertools
import os

bitdefender_file = 'subory/bitdefender.txt'
kaspersky_file = 'subory/kaspersky.txt'
mcafee_file = 'subory/mcafee.txt'
eset_file = 'subory/eset.txt'
output_file = 'subory/same.txt'
antivirusy = {0, 1, 2, 3}  # mnozina o velkosti poctu AV na tvorbu permutacii
rovnake = []
files = []


def intersections():
    # vrati triedy ktore sa prelinaju v danom pocte (podmnoziny) AV
    with open(bitdefender_file, 'r') as file1, open(kaspersky_file, 'r') as file2, \
            open(mcafee_file, 'r') as file3, open(eset_file, 'r') as file4:
        files.extend((file1, file2, file3, file4))
        # same = set(file2).intersection(file1, file3)
        podmnoziny = list(itertools.combinations(antivirusy, 3))
        for i in range(len(podmnoziny)):
            # rovnake.append(set(files[podmnoziny[i][0]]).intersection(files[podmnoziny[i][1]]))
            rovnake.append(set(files[podmnoziny[i][0]]).intersection(files[podmnoziny[i][1]],
                                                                     files[podmnoziny[i][2]]))
            # same1 = set(file2).intersection(file1)

    for i in range(1, len(rovnake)):
        rovnake[0] = rovnake[0].union(rovnake[i])
    with open(output_file, 'w') as out:
        for name in rovnake[0]:
            out.write(name)


# intersections()