import itertools
import os

antivirusy = {0, 1, 2, 3}
rovnake = []
files = []
# vrati triedy ktore sa prelinaju v danom pocte (podmnoziny) AV
with open('subory/bitdefender.txt', 'r') as file1:
    with open('subory/kaspersky.txt', 'r') as file2:
        with open('subory/mcafee.txt', 'r') as file3:
            with open('subory/eset.txt', 'r') as file4:
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
out = open('subory/same.txt', 'w')
for name in rovnake[0]:
    # print(name[:-1])
    out.write(name)

for file in files:
    # os.path.basename(f.name)
    file.close()
out.close()
