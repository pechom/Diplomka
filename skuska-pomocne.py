def counts():
    featues = Counter()
    for line in lines:
        for key in keys:
            if key in line:
                featues[key] = featues[key] + 1
    return featues.items()
#----------------------------

def header_byte_entropy():
    colnames = []
    st = ['mean', 'var', 'median', 'max', 'min', 'max-min']
    colnames.extend(['ent_q_diffs_' + str(x) for x in range(21)])
    colnames.extend(['ent_q_diffs_' + x for x in st])
    return colnames

def header_asm_sections():
    kown_sections = ['.text', '.data', '.bss', '.rdata', '.edata', '.idata', '.rsrc', '.tls', '.reloc']
    colnames = kown_sections + ['Num_Sections', 'Unknown_Sections', 'Unknown_Sections_lines']
    colnames += ['.text_por', '.data_por', '.bss_por', '.rdata_por', '.edata_por',
                 '.idata_por', '.rsrc_por', '.tls_por', '.reloc_por']
    colnames += ['known_Sections_por', 'Unknown_Sections_por', 'Unknown_Sections_lines_por']
    return colnames

# zapis do csv
meta_data_colnames = header_asm_meta_data()
sym_colnames = header_asm_sym()
meta_data_csv_w = writer(meta_data_csv)
meta_data_csv_w.writerows([meta_data_colnames])
# a potom sa postupne vpisuju stlpce
registers_csv_w.writerows([registers])


# three different types: memory, constant, register
# memory: dword, word, byte
# constant: arg, var
# register: eax ebx ecx edx esi edi esp ebp ax bx cx dx ah bh ch dh al bl cl dl

def get_pattern(lst):
    # return a pattern for a length 2 list
    if len(lst) == 2:
        first = lst[0]
        tmp = lst[-1].split(', ')
        if len(tmp) == 2:
            second, third = id_pattern(tmp[0]), id_pattern(tmp[1])
            return first + '_' + second + '_' + third
    return None


def id_pattern(s):
    # given a string return its type (memory, constant, register,number, other)
    if any(m in s for m in ['dword', 'word', 'byte']):
        return 'memory'
    elif any(r in s for r in
             ['ax', 'bx', 'cx', 'dx', 'ah', 'bh', 'ch', 'dh', 'al', 'bl', 'cl', 'dl', 'esi', 'edi', 'esp', 'ebp']):
        return 'register'
    elif any(r in s for r in ['arg', 'var']):
        return 'constant'
    elif is_hex(s):
        return 'number'
    else:
        return 'other'
#----------------------

# generate grams dictionary for one file
def grams_dict(f_name, N=4):
    path = "train/%s.bytes" % f_name
    one_list = []
    with open(path, 'rb') as f:
        for line in f:
            one_list += line.rstrip().split(" ")[1:]
    grams_string = [''.join(one_list[i:i + N]) for i in range(len(one_list) - N + 1)]
    tree = dict()
    for gram in grams_string:
        if gram not in tree:
            tree[gram] = 1
    return tree


# add up ngram dictionaries
def reduce_dict(f_labels):
    result = dict()
    for f_name in f_labels:
        d = grams_dict(f_name)
        for k, v in d.iteritems():
            if k in result:
                result[k] += v
            else:
                result[k] = v
        del d
    # print "this class has %i keys"%len(result)
    # pickle.dump(result, open('gram/ngram_%i'%label,'wb'))
    return result

# document frequency
def gen_df(features_all, train=True, verbose=False, N=4):
    yield ['Id'] + features_all  # yield header
    if train == True:
        ds = 'train'
    else:
        ds = 'test'
    directory_names = list(set(glob.glob(os.path.join(ds, "*.bytes"))))
    for f in directory_names:
        f_id = f.split('/')[-1].split('.')[0]
        if verbose == True:
            print
            'doing %s' % f_id
        one_list = []
        with open("%s/%s.bytes" % (ds, f_id), 'rb') as read_file:
            for line in read_file:
                one_list += line.rstrip().split(" ")[1:]
        grams_string = [''.join(one_list[i:i + N]) for i in range(len(one_list) - N)]
        # build a dict for looking up

        grams_dict = dict()
        for gram in grams_string:
            if gram not in grams_dict:
                grams_dict[gram] = 1

        binary_features = []
        for feature in features_all:
            if feature in grams_dict:
                binary_features.append(1)
            else:
                binary_features.append(0)
        del grams_string
        '''
        ## instead of binary features, do count
        grams_dict = dict()
        for gram in grams_string:
            if gram not in grams_dict:
                grams_dict[gram] = 1
            else:
                grams_dict[gram] += 1 

        binary_features = []
        for feature in features_all:
            if feature in grams_dict:
                binary_features.append(grams_dict[feature])
            else:
                binary_features.append(0)
        del grams_string        
        '''
        yield [f_id] + binary_features

#----------------

# instruction frequency
instr_set = set(['mov'])
def consolidate(path, instr_set=instr_set):
    Files = os.listdir(path)
    asmFiles = [i for i in Files if '.asm' in i]
    consolidatedFile = path + '_instr_frequency.csv'
    with open(consolidatedFile, 'wb') as f:
        fieldnames = ['Id'] + list(instr_set)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t, fname in enumerate(asmFiles):
            consolidation = dict(zip(instr_set, [0] * len(instr_set)))
            consolidation['Id'] = fname[:fname.find('.asm')]
            with open(path + '/' + fname, 'rb') as f:
                for line in f:
                    if 'text' in line and ',' in line and ';' not in line:
                        row = line.lower().strip().split('  ')[1:]
                        if row:
                            tmp_list = [x.strip() for x in row if x != '']
                            if len(tmp_list) == 2 and tmp_list[0] in consolidation:
                                consolidation[tmp_list[0]] += 1
            writer.writerow(consolidation)

#--------------------------------
# sizes
X_test[i][0] = os.path.getsize('test/' + Id + '.asm')
X_test[i][1] = os.path.getsize('test/' + Id + '.bytes')

#----------------------
# strings bins
def extract_length(data):
    another_f = np.vstack([x[2] for x in data])

    len_arrays = [np.array([len(y) for y in x[1]] + [0] + [10000]) for x in data]
    bincounts = [np.bincount(arr) for arr in len_arrays]

    counts = np.concatenate([another_f[:, :3], np.vstack([arr[4:100] for arr in bincounts])], axis=1)
    counts_0_10 = np.sum(counts[:, 0:10], axis=1)[:, None]                                                      #!!!!!!!!!
    med = np.array([np.median([len(y) for y in x[1]] + [0]) for x in data])[:, None]
    mean = np.array([np.mean([len(y) for y in x[1]] + [0]) for x in data])[:, None]
    var = np.array([np.var([len(y) for y in x[1]] + [0]) for x in data])[:, None]

def dump_names(strings_feats_dir):
    n = ['string_len_counts_' + str(x) for x in range(1, 100)] + [
        'string_len_counts_0_10']
hickle.dump(n, os.path.join(strings_feats_dir, 'strings_feats_names'))
