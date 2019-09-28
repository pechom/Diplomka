def readdata(name):
    '''Read the data from a file and count how often each byte value
    occurs.'''
    f = open(name, 'rb')
    data = f.read()
    f.close()
    ba = bytearray(data)
    del data
    counts = [0]*256
    for b in ba:
        counts[b] += 1
    return (counts, float(len(ba)))

def entropy(counts, size): # pocty jednotlivych bytov a velkost suboru z ktoreho sa citalo - rovnako pre znaky
    '''Calculate the entropy of the data represented by the counts list'''
    ent = 0.0
    print ("size " + str(size))
    for b in counts:
        if b == 0:
            continue
        p = float(b)/size
        ent -= p*math.log(p, 256) #alebo 2 namiesto 256 a nemusim *8
    return ent*8

#----------------------------------------------
#entropy - 2016-6 = [27] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def H1(data, block_size, step, counts, ent): # entropy of block
    entropy = 0.0
    # counts = np.zeros(17,dtype=np.float)
    for i in range(block_size):
        counts[data[i]] += 1

    for i in range(counts.shape[0]):
        if counts[i] > 0:
            entropy += - counts[i] / block_size * \
                       np.log2(counts[i] / block_size)

    # cb0 = counts[np.where(counts>0)]

    # entropy = np.sum(-cb0/block_size * np.log2(cb0/block_size))
    ent[0] = entropy

    for i in range(1, data.shape[0] - block_size):

        dec = counts[data[i - 1]]
        inc = counts[data[i + block_size - 1]]

        counts[data[i - 1]] -= 1
        counts[data[i + block_size - 1]] += 1

        entropy -= -dec / block_size * np.log2(dec / block_size)
        if dec > 1:
            entropy += -(dec - 1) / block_size * \
                       np.log2((dec - 1) / block_size)

        if inc > 0:
            entropy -= -inc / block_size * np.log2(inc / block_size)

        entropy += - (inc + 1) / block_size * np.log2((inc + 1) / block_size)

        if i % step == 0:
            ent[i / step] = (entropy)
H_numba = autojit(H1, nopython=True)

def get_entropy_features(byte_data): # entropia bytov
    corr = {str(key): key for key in range(10)}
    corrl = {'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, '?': 16}
    corr.update(corrl)

    block_size = 10000
    step = 100
    text = byte_data
    # name = filename.split('/')[-1].split('.')[0]

    # with gzip.open(filename, 'r') as f:
    #    text = f.read()

    lines = ''.join(byte_data).split('\r\n')

    t = []
    for l in lines:
        elems = l.split(' ')
        t.extend(elems[1:])
    t = ''.join(t)

    chararray = np.array([corr[x] for x in t])

    counts = np.zeros(17, dtype=np.float)
    ent = np.zeros((chararray.shape[0] - block_size) / step + 1)
    H_numba(chararray, block_size, step, counts, ent)

    # return [name, ent]
    return ent
n.extend(['ent_p_' + str(x) for x in range(20)])
hickle.dump(n, os.path.join(ent_feats_dir, 'ent_feats_names'))

#--------------------------------------
# extracted strings entropy
# map printable characters 0x20 - 0x7f to an int array consisting of 0-95, inclusive
as_shifted_string = [b - ord(b'\x20') for b in b''.join(allstrings)]
c = np.bincount(as_shifted_string, minlength=96)  # histogram count
# distribution of characters in printable strings
csum = c.sum()
p = c.astype(np.float32) / csum
wh = np.where(c)[0]
H = np.sum(-p[wh] * np.log2(p[wh]))  # entropy
else:
avlength = 0
c = np.zeros((96,), dtype=np.float32)
H = 0
csum = 0

#-----------------------------------------
class ByteEntropyHistogram(FeatureType):
    ''' 2d byte/entropy histogram based loosely on (Saxe and Berlin, 2015).
    This roughly approximates the joint probability of byte value and local entropy.
    See Section 2.1.1 in https://arxiv.org/pdf/1508.03096.pdf for more info.
    '''

    name = 'byteentropy'
    dim = 256

    def __init__(self, step=1024, window=2048):
        super(FeatureType, self).__init__()
        self.window = window
        self.step = step

    def _entropy_bin_counts(self, block):
        # coarse histogram, 16 bytes per bin
        c = np.bincount(block >> 4, minlength=16)  # 16-bin histogram
        p = c.astype(np.float32) / self.window
        wh = np.where(c)[0]
        H = np.sum(-p[wh] * np.log2(
            p[wh])) * 2  # * x2 b.c. we reduced information by half: 256 bins (8 bits) to 16 bins (4 bits)

        Hbin = int(H * 2)  # up to 16 bins (max entropy is 8 bits)
        if Hbin == 16:  # handle entropy = 8.0 bits
            Hbin = 15

        return Hbin, c

    def raw_features(self, bytez, lief_binary):
        output = np.zeros((16, 16), dtype=np.int)
        a = np.frombuffer(bytez, dtype=np.uint8)
        if a.shape[0] < self.window:
            Hbin, c = self._entropy_bin_counts(a)
            output[Hbin, :] += c
        else:
            # strided trick from here: http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html
            shape = a.shape[:-1] + (a.shape[-1] - self.window + 1, self.window)
            strides = a.strides + (a.strides[-1],)
            blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::self.step, :]

            # from the blocks, compute histogram
            for block in blocks:
                Hbin, c = self._entropy_bin_counts(block)
                output[Hbin, :] += c

        return output.flatten().tolist()

    def process_raw_features(self, raw_obj):
        counts = np.array(raw_obj, dtype=np.float32)
        sum = counts.sum()
        normalized = counts / sum
        return normalized

class ByteHistogram(FeatureType):
    ''' Byte histogram (count + non-normalized) over the entire binary file '''

    name = 'histogram'
    dim = 256

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_binary):
        counts = np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256)
        return counts.tolist()

    def process_raw_features(self, raw_obj):
        counts = np.array(raw_obj, dtype=np.float32)
        sum = counts.sum()
        normalized = counts / sum
        return normalized

#---------------------  z netu
def entropy(string):
"Calculates the Shannon entropy of a string"
     # get probability of chars in string
    prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]

     # calculate the entropy
    entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])


def entropy(bytes):
    (float)
    entropy = 0
    for i in counts
    do
(float)p = Counts[i] / filesize
if (p > 0) entropy = entropy - p * lg(p) # log2


def entropy_normalized(bytes)
    for count in byte_counts:
        # If no bytes of this value were seen in the value, it doesn't affect
        # the entropy of the file.
        if count == 0:
            continue
        # p is the probability of seeing this byte in the file, as a floating-
        # point number
        p = 1.0 * count / total
        entropy -= p * math.log(p, 256)