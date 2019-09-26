#!/bin/bash
#
# Given a set of word embeddings as the 1st argument,
# runs Relation Extraction and Sentiment Analysis
# experiments on those embeddings and saves to a log file

EMBEDDINGS=$1
LOGFILE=$2
PYTHON=$3

if [ -z "${EMBEDDINGS}" ]; then
    cat <<EOF
Usage: $0 EMBEDDINGS [LOGFILE [PYTHON]]
  EMBEDDINGS  Path to a .txt word embeddings file
  LOGFILE     (Optional) File to log experiment outcomes to; if not
              provided, uses basename of EMBEDDINGS
  PYTHON      (Optional) Path to Python binary to execute (Requires >=3)
              If not provided, defaults to python3
EOF
    exit
fi

if [ -z "${LOGFILE}" ]; then
    LOGFILE=$(basename ${EMBEDDINGS}).log
fi
if [ -z "${PYTHON}" ]; then
    PYTHON=python3
fi

echo "=== Relation extraction ===" > ${LOGFILE}
${PYTHON} -m Relation_extraction.train_cnn \
    ${EMBEDDINGS} \
    Relation_extraction/dataset \
    >> ${LOGFILE}

echo "=== Sentiment analysis ===" >> ${LOGFILE}
${PYTHON} -m sentiment_classification.train \
    ${EMBEDDINGS} \
    >> ${LOGFILE}
