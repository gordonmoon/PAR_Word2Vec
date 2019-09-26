'''
'''

def loadFull(f):
    data = []
    with open(f, 'r') as stream:
        # skip headers
        for _ in range(11): stream.readline()
        # load data
        for line in stream:
            (_, w1, w2, score) = [s.strip() for s in line.split('\t')]
            data.append((w1, w2, float(score)))
    return data

def loadOne(f):
    data = []
    with open(f, 'r') as stream:
        for line in stream:
            (w1, w2, score) = [s.strip() for s in line.split('\t')]
            data.append((w1, w2, float(score)))
    return data

def load(config):
    full = loadFull(config['WordSim-353']['Dataset_File'])
    sim = loadOne(config['WordSim-353']['Similarity_File'])
    rel = loadOne(config['WordSim-353']['Relatedness_File'])
    return (full, sim, rel)
