from vt import VT
vt = VT()

#https://github.com/nu11p0inter/virustotal-api-v2
# key management
vt.setkey('0c4047d034a7f8e0f5afd8884d47af625df14618c8ec5c98ff03d599a8f5441e')

vt.scanfile('C:\CFFExplorer.exe')
print(vt.out('json'))
