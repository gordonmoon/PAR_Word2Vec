import numpy as np

def load_embeddings(embeddings_file, words=None):
    wordEmbeddings = {}
    word2Idx = {}

    if words and len(words) > 0:
        word_filter = lambda w: w.lower() in words
    else:
        word_filter = lambda w: True

    # :: Load the pre-trained embeddings file ::
    fEmbeddings = open(embeddings_file, 'r')

    print("Load pre-trained embeddings file")
    for line in fEmbeddings:
        split = line.strip().split(" ")
        if len(split) == 2:
            continue
        word = split[0]

        if len(word2Idx) == 0: #Add padding+unknown
            word2Idx["PADDING_TOKEN"] = len(word2Idx)
            vector = np.zeros(len(split)-1) #Zero vector vor 'PADDING' word
            wordEmbeddings["PADDING_TOKEN"] = vector

            word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, len(split)-1)
            wordEmbeddings["UNKNOWN_TOKEN"] = vector

        if word_filter(word):
            vector = np.array([float(num) for num in split[1:]])
            wordEmbeddings[word] = vector
            word2Idx[word] = len(word2Idx)

    return (wordEmbeddings, word2Idx)

def load_embeddings_matrix(embeddings_file, words=None):
    wordEmbeddings, word2Idx = load_embeddings(embeddings_file, words=words)

    wordEmbeddings = np.array([
        wordEmbeddings[word]
            for (word, _) in sorted(word2Idx.items(), key=lambda k:k[1])
    ])

    return wordEmbeddings, word2Idx

def load_embeddings_dict(embeddings_file, words=None):
    wordEmbeddings, word2Idx = load_embeddings(embeddings_file, words=words)

    return wordEmbeddings
