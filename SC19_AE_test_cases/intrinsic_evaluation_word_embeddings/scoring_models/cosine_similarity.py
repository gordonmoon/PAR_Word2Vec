import numpy as np
from dependencies import pyemblib
from dependencies.drgriffis.common import log

class CosineSimilarityScorer:
    
    def __init__(self, embf, embmode, log=log):
        log.writeln('Reading embeddings from %s...' % embf)
        embs = pyemblib.read(embf, mode=embmode, replace_errors=True)
        log.writeln('  Read %d embeddings.' % len(embs))
        self._embs = embs

    def score(self, w1, w2):
        w1_v = self._embs.get(w1.lower(), None)
        w2_v = self._embs.get(w2.lower(), None)

        if w1_v is None or w2_v is None:
            return None
        else:
            w1_v, w2_v = np.array(w1_v), np.array(w2_v)
            numerator = np.dot(w1_v, w2_v)
            denominator = np.linalg.norm(w1_v) * np.linalg.norm(w2_v)
            return numerator/denominator

