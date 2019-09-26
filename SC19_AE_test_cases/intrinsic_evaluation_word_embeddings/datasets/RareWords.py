'''
'''

def load(config):
    data = []
    with open(config['RareWords']['Dataset_File'], 'r') as stream:
        for line in stream:
            (w1, w2, score, _) = [s.strip() for s in line.split('\t', 3)]
            data.append((w1, w2, float(score)))
    return data
