from vt import VT
import requests

vt = VT()

# https://github.com/nu11p0inter/virustotal-api-v2
# key management
# vt.setkey('0c4047d034a7f8e0f5afd8884d47af625df14618c8ec5c98ff03d599a8f5441e')
# vt.out('json')
# scan = vt.scanfile('D:\AMDPRW.msi')
# print(scan)

url = 'https://www.virustotal.com/vtapi/v2/file/report'

params = {'apikey': '0c4047d034a7f8e0f5afd8884d47af625df14618c8ec5c98ff03d599a8f5441e',
          'resource': '029c84a52e2503fd549180742c25eb863c0e7a46'}

response = requests.get(url, params=params)

print(response.json())
