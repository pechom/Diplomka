import pefile
import os
import sys

#mal_file = sys.argv[1]
#pe = pefile.PE(mal_file)
pe = pefile.PE('C:/Program Files/NTCore/Explorer Suite/CFF Explorer.exe')
#print(pe.dump_info())
#if open('C:/Program Files/NTCore/Explorer Suite/CFF Explorer.exe').read(2) == "MZ":
if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
    for entry in pe.DIRECTORY_ENTRY_IMPORT:
        print ("%s" % entry.dll)
    for imp in entry.imports:
        if imp.name != None:
            print ("\t%s" % (imp.name))
        else:
            print ("\tord(%s)" % (str(imp.ordinal)))
print("\n")

if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT'):
    for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
        print ("%s" % exp.name)
print("\n")

for section in pe.sections:
    print("%s %s %s %s" % (section.Name, hex(section.VirtualAddress), hex(section.Misc_VirtualSize), section.SizeOfRawData), section.get_hash_md5())
print("\n")

print(pe.get_imphash())
print("\n")

strings = os.popen("strings '{0}'".format('C:/Program Files/NTCore/Explorer Suite/CFF Explorer.exe')).read()
strings = set(strings.split("\n"))
for string in strings:
    print(string)
print(len(strings))
print("\n")


# python pefileTest.py >> pefile.txt
