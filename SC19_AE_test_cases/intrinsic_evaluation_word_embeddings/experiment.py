'''
'''

import os
import codecs
from types import SimpleNamespace
import configparser
from scipy.stats import spearmanr
import numpy as np
from datasets import *
from scoring_models import *
import detailed_logging
from dependencies import configlogger
from dependencies import pyemblib
from dependencies.drgriffis.common import log

WS353 = 'WordSim-353'
# WS353_SIM = 'WS353_Similarity'
# WS353_REL = 'WS353_Relatedness'
SL999 = 'SimLex-999'
# RW = 'RareWords'

def assignScores(dataset, scorer, detailed_log=None, ds_name=None):
    paired_scores, skipped = [], 0
    kept_samples, gold, pred = [], [], []
    skipped = 0
    for (w1, w2, gold_score) in dataset:
        pred_score = scorer.score(w1, w2)
        if pred_score is None:
            skipped += 1
        else:
            kept_samples.append((w1, w2, gold_score))
            gold.append(gold_score)
            pred.append(pred_score)

    if detailed_log:
        detailed_logging.logPredictions(kept_samples, pred, gold, ds_name, detailed_log)

    (rho, _) = spearmanr(gold, pred)

    metrics = SimpleNamespace()
    metrics.rho = rho
    metrics.evaluated = len(kept_samples)
    metrics.total = len(dataset)
    
    return metrics

def experiment(config, scorer_class, *args, detailed_log=None):
    (ws_full, ws_sim, ws_rel) = WordSim353.load(config)
    sl = SimLex999.load(config)
    # rw = RareWords.load(config)

    scorer = scorer_class(*args)
    results = {}

    for (dat, lbl) in [
                (ws_full, WS353),
                # (ws_sim, WS353_SIM),
                # (ws_rel, WS353_REL),
                (sl, SL999),
                # (rw, RW)
            ]:
        log.writeln('\nChecking %s...' % lbl)
        ds_metrics = assignScores(
            dat,
            scorer,
            detailed_log=detailed_log,
            ds_name=lbl
        )

        results[lbl] = ds_metrics
        log.writeln('Rho: %.2f (Evaluated %d/%d)' % (
            ds_metrics.rho, ds_metrics.evaluated, ds_metrics.total
        ))

    log.writeln('\n\n-- Final report --\n')
    # for lbl in [WS353, WS353_SIM, WS353_REL, SL999, RW]:
    for lbl in [WS353, SL999]:
        m = results[lbl]
        log.writeln('%s --> Rho: %.3f (%d/%d samples)' % (
            lbl, m.rho, m.evaluated, m.total
        ))

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog EMBEDDING_FILE')
        parser.add_option('--config', dest='configf',
                default='config.ini',
                help='experiment configuration file (default %default)')
        parser.add_option('--mode', dest='emb_mode',
                type='choice', choices=[pyemblib.Mode.Text, pyemblib.Mode.Binary],
                default=pyemblib.Mode.Binary,
                help='embedding file format (%s or %s)' % (pyemblib.Mode.Text, pyemblib.Mode.Binary))
        parser.add_option('--detailed-scores', dest='detailed_scoresf',
                default=None,
                help='file to write detailed scoring results to')
        parser.add_option('-l', '--logfile', dest='logfile',
                help='name of file to write log contents to (empty for stdout)',
                default=None)
        (options, args) = parser.parse_args()

        if not os.path.exists(options.configf):
            raise ValueError('Configuration file "%s" not found!' % options.configf)

        if len(args) != 1:
            parser.print_help()
            exit()
        return args, options

    (embf,), options = _cli()
    log.start(logfile=options.logfile)

    configlogger.writeConfig(log, [
        ('Embedding file', embf),
        ('Embedding file format', options.emb_mode),
        ('Configuration file', options.configf),
        ('Logging detailed scores to', options.detailed_scoresf)
    ], 'Similarity/relatedness experiment')

    config = configparser.ConfigParser()
    config.read(options.configf)

    if options.detailed_scoresf:
        detailed_log = codecs.open(options.detailed_scoresf, 'w', 'utf-8')
    else:
        detailed_log = None

    experiment(
        config,
        CosineSimilarityScorer,
        embf,
        options.emb_mode,
        detailed_log=detailed_log
    )

    if detailed_log:
        detailed_log.close()

    log.stop()
