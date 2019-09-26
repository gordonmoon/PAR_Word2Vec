#!/usr/bin/env python

import sys
import codecs
import re

def getStats(logf, one_line=False):
    paired = {
        'Relation extraction' : r'Non-other Macro-Averaged F1:',
        'Sentence polarity classification': 'Test-Accuracy',
        'Sentiment analysis': 'Test accuracy',
        'Subjectivity classification': 'Test-Accuracy',
        'SNLI': 'Test loss'
    }

    buff = []
    cur_task = None
    with codecs.open(logf, 'r', 'utf-8') as stream:
        for line in stream:
            if line[:4] == '=== ':
                cur_task = line.strip(' =\n')
            else:
                if cur_task in paired:
                    search = paired[cur_task]
                    if re.match(search, line):
                        if not one_line:
                            buff.append(cur_task)
                            buff.append(line)
                        else:
                            buff.append(
                                line.split(':')[1].split()[0]
                            )
    return buff

logf = sys.argv[1]
one_line = False
if len(sys.argv) > 2:
    flag = sys.argv[2]
    if flag == "--one-line":
        one_line = True

buff = getStats(logf, one_line=one_line)

if one_line:
    print(', '.join(buff))
else:
    for line in buff:
        print(line)
