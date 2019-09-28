import Levenshtein

f1 = open("subory/microsoft_names.txt", "r")
f2 = open("subory/kaspersky_names.txt", "r")

list1 = f1.readlines()
list2 = f2.readlines()

for line1 in list1:
    for line2 in list2:
        # if line1 == line2:
        if Levenshtein.distance(line1, line2) < 3:
            print(line1[:-1])
f1.close()
