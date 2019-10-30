import Levenshtein

f1 = open("stare subory/microsoft_names.txt", "r")
f2 = open("stare subory/kaspersky_names.txt", "r")

list1 = f1.readlines()
list2 = f2.readlines()
f1.close()
f2.close()

for line1 in list1:
    for line2 in list2:
        if 3 > Levenshtein.distance(line1, line2) > 1:
            print(line1[:-1])  # posledny je znak konca riadku
            print(line2)
