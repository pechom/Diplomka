import requests, sys, json, hashlib

# assigns the values passed in at the command line to variables
user_api_key = '0c4047d034a7f8e0f5afd8884d47af625df14618c8ec5c98ff03d599a8f5441e'
file = str(sys.argv[1])

def sha256sum(filename):
    """
    Efficient sha256 checksum realization
    Take in 8192 bytes each time
    The block size of sha256 is 512 bytes
    """
    with open(filename, 'rb') as f:
        m = hashlib.sha256()
        while True:
            data = f.read(8192)
            if not data:
                break
            m.update(data)
        return m.hexdigest()

params = {
    'apikey': user_api_key,
    'resource': sha256sum(file)
    }


# makes json pretty
def pp_json(json_thing, sort=True, indents=4):
    if type(json_thing) is str:
        print(json.dumps(json.loads(json_thing), sort_keys=sort, indent=indents))
    else:
        print(json.dumps(json_thing, sort_keys=sort, indent=indents))
        return None


response = requests.get('https://www.virustotal.com/vtapi/v2/file/report', params=params)

# turn the respose into json
json_response = response.json()


# make it pretty
pretty_json = pp_json(json_response)

print(pretty_json)

# python vtscript.py C:\CFFExplorer.exe >> virustotal.json