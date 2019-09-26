'''
'''

def load(config):
    data = []
    with open(config['SimLex-999']['Dataset_File'], 'r') as stream:
        # skip headers
        stream.readline()
        # load data
        for line in stream:
            (w1, w2, _, score, _) = [s.strip() for s in line.split('\t', 4)]
            data.append((w1, w2, float(score)))
    return data
