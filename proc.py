# Pre-processing module for the language models

import numpy as np
import os
import lm_tools
from nltk.tokenize import wordpunct_tokenize
from collections import Counter
from collections import defaultdict
from scipy.sparse import lil_matrix, sparsetools


def process():
    """
    Specify the following:
    """
    ##################################
    train_captions = 'data/iaprtc12/train_captions.txt'
    train_images = 'data/iaprtc12/train_hidden7.txt'
    test_captions = 'data/iaprtc12/test_captions.txt'
    test_images = 'data/iaprtc12/test_hidden7.txt'
    context = 5
    ##################################

    # Load captions
    print 'Loading captions...'
    train = load_captions(train_captions)
    test = load_captions(test_captions)

    # Tokenize the data
    print 'Tokenizing...'
    train_tokens = tokenize(train, context=context)
    test_tokens = tokenize(test, context=context)

    # Index words and create vocabulary
    print 'Creating vocabulary...'
    (word_dict, index_dict) = index_words(train_tokens)
    
    # Compute n-grams
    print 'Computing n-grams...'
    train_ngrams = lm_tools.get_ngrams(train_tokens, context=context)
    test_ngrams = lm_tools.get_ngrams(test_tokens, context=context)

    # Compute sparse label matrix
    print 'Computing labels...'
    labels = compute_labels(train_ngrams, word_dict, context=context)

    # Compute model instances
    print 'Computing model instances...'
    (train_instances, train_index) = lm_tools.model_inputs(train_ngrams, word_dict,
        context=context, include_last=False, include_index=True)
    (test_instances, test_index) = lm_tools.model_inputs(test_ngrams, word_dict,
        context=context, include_last=False, include_index=True)

    # Load image features
    print 'Loading image features...'
    trainIM = load_convfeatures(train_images)
    testIM = load_convfeatures(test_images)

    # Save everything into dictionaries
    print 'Packing up...'
    z = {}
    z['text'] = train
    z['tokens'] = train_tokens
    z['word_dict'] = word_dict
    z['index_dict'] = index_dict
    z['ngrams'] = train_ngrams
    z['labels'] = labels
    z['instances'] = train_instances
    z['IM'] = trainIM
    z['index'] = train_index
    z['context'] = context

    zt = {}
    zt['text'] = test
    zt['tokens'] = test_tokens
    zt['ngrams'] = test_ngrams
    zt['instances'] = test_instances
    zt['IM'] = testIM
    zt['index'] = test_index
    zt['context'] = context

    return (z, zt)
    

def load_convfeatures(loc):
    """
    Reads in the txt file produces by ConvNet
    Consider modifying this for other file types (e.g .npy)
    """
    return np.loadtxt(loc)
    

def load_captions(loc):
    """
    Load the captions
    """
    f = open(loc, 'rb')
    captions = []
    for line in f:
        captions.append(line.strip())
    return captions


def is_number(s):
    try:
        float(s)
        return '<#>'
    except ValueError:
        return s


def normalize(pairs):
    """
    Normalize rows of a csr matrix (sum to 1)
    """
    factor = pairs.sum(axis=1)
    nnzeros = np.where(factor > 0)
    factor[nnzeros] = 1 / factor[nnzeros]
    factor = np.array(factor)[0]
    if not pairs.format == "csr":
         raise ValueError("csr only")
    sparsetools.csr_scale_rows(pairs.shape[0], pairs.shape[1], pairs.indptr,
        pairs.indices, pairs.data, factor)
    return pairs


def uniq(seq):
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if x not in seen and not seen_add(x)]


def tokenize(X, context=5, start='<start>', end='<end>'):
    """
    Tokenize each of the captions
    """
    tokens = [wordpunct_tokenize(x) for x in X]
    tokens = [ [w.lower() for w in x] for x in tokens ]
    tokens = [ [is_number(w) for w in x] for x in tokens ]
    for i, x in enumerate(tokens):
        tokens[i] = [start] * context + x + [end]
    return tokens


def get_counts(tokens):
    """
    Compute a dictionary of counts from tokens
    """
    flat_tokens = [item for sublist in tokens for item in sublist]
    word_counts = Counter(flat_tokens)
    return word_counts


def index_words(tokens):
    """
    Compute dictionaries for indexing words
    """
    flat_tokens = [item for sublist in tokens for item in sublist]
    word_dict = {}
    for i, w in enumerate(list(set(flat_tokens))):
        word_dict[w] = i
    word_dict['unk'] = i+1
    index_dict = dict((v,k) for k, v in word_dict.iteritems())
    return (word_dict, index_dict)


def compute_labels(ngrams, word_dict, context=5):
    """
    Create matrix of word occurences (labels for the model)
    """
    ngrams_count = [len(x) for x in ngrams]
    uniq_ngrams = uniq([item[:-1] for sublist in ngrams for item in sublist])
    count = 0
    train_dict = {}
    for w in uniq_ngrams:
        train_dict[w] = count
        count = count + 1

    labels = lil_matrix((sum(ngrams_count), len(word_dict.keys())))
    train_ngrams_flat = [item for sublist in ngrams for item in sublist]
    labels_dict = defaultdict(int)
    col_dict = defaultdict(list)

    for w in train_ngrams_flat:
        row_ind = train_dict[w[:context]]
        col_ind = word_dict[w[-1]]
        labels_dict[(row_ind, col_ind)] += 1
        col_dict[row_ind] = list(set(col_dict[row_ind] + [col_ind]))

    count = 0
    for x in ngrams:
        for w in x:
            row_ind = train_dict[w[:context]]
            inds = col_dict[(row_ind)]
            labels[count, word_dict[w[-1]]] = 1
            count = count + 1

    labels_un = labels.tocsr()
    labels = normalize(labels_un)
    return labels


