# Language model tools for LBL and MLBL

import numpy as np
import bleu
import copy
from collections import defaultdict
from scipy.linalg import norm


def compute_ngrams(sequence, n):
    """
    Return n-grams from the input sequence
    """
    sequence = list(sequence)
    count = max(0, len(sequence) - n + 1)
    return [tuple(sequence[i:i+n]) for i in range(count)]


def get_ngrams(X, context=5):
    """
    Extract n-grams from each caption in X
    """
    ngrams = []
    for x in X:
        x_ngrams = compute_ngrams(x, context + 1)
        ngrams.append(x_ngrams)
    return ngrams


def load_embeddings(loc = 'embeddings/embeddings-scaled.EMBEDDING_SIZE=50.txt'):
    """
    Load pre-trained word embeddings
    """
    embed_map = {}
    ap = open(loc, 'r')
    for line in ap:
        entry = line.split(' ')
        key = entry[0]
        value = [float(x) for x in entry[1:]]
        embed_map[key] = value
    return embed_map


def model_inputs(ngrams, word_dict, context=5, include_last=True, include_index=False):
    """
    Maps ngrams to format used for the language model
    include_last=True for evaluation (LL, perplexity)
    Out of vocabulary words are mapped to 'unk' (unknown) token
    """
    d = defaultdict(lambda : 0)
    for w in word_dict.keys():
        d[w] = 1   
    ngrams_count = [len(x) for x in ngrams]
    if include_last:
        instances = np.zeros((sum(ngrams_count), context + 1))
    else:
        instances = np.zeros((sum(ngrams_count), context))
    count = 0
    index = np.zeros((sum(ngrams_count), 1))
    for i in range(len(ngrams)):
        for j in range(len(ngrams[i])):
            values = [word_dict[w] if d[w] > 0 else word_dict['unk']
                for w in ngrams[i][j]]
            if include_last:
                instances[count] = values
            else:
                instances[count] = values[:-1]
            index[count] = i
            count = count + 1
    instances = instances.astype(int)
    if include_index:
        return (instances, index)
    else:
        return instances


def compute_ll(net, instances, Im=None):
    """
    Compute the log-likelihood of instances from net
    """
    if Im != None:
        preds = net.forward(instances[:,:-1], Im)[-1]
    else:
        preds = net.forward(instances[:,:-1])[-1]
    ll = 0
    for i in range(preds.shape[0]):
        ll += np.log2(preds[i, instances[i, -1]] + 1e-20)
    return ll
    

def perplexity(net, ngrams, word_dict, Im=None, context=5):
    """
    Compute the perplexity of ngrams from net
    """
    ll = 0
    N = 0
    for i, ng in enumerate(ngrams):
        instances = model_inputs([ng], word_dict)
        if Im != None:
            ll += compute_ll(net, instances, np.tile(Im[i], (len(ng), 1)))
        else:
            ll += compute_ll(net, instances)
        N += len(instances)
    return pow(2, (-1.0 / N) * ll)


def weighted_sample(n_picks, weights):
    """
    Sample from a distribution weighted by 'weights'
    """
    t = np.cumsum(weights)
    s = np.sum(weights)
    return np.searchsorted(t, np.random.rand(n_picks) * s)


def sample(net, word_dict, index_dict, num, Im=None, initial=None, use_end=False):
    """
    Sample from the model
    use_end: if an <end> token is sampled, quit. Otherwise do not include them.
    """
    if initial == None:
        initial = ['<start>'] * net.context
    inputs = np.array([word_dict[w] for w in initial]).reshape(1, net.context)
    done = False
    count = 0
    while not done:
        if Im != None:
            preds = net.forward(inputs[:,inputs.shape[1]-net.context:], [Im])[-1]
        else:
            preds = net.forward(inputs[:,inputs.shape[1]-net.context:])[-1]
        pw = weighted_sample(1, preds[:,:-1].flatten())
        token = index_dict[int(pw)]
        if token == '<end>':
            if use_end:
                done = True
            else:
                preds[:, int(pw)] = 0.0
                pw = weighted_sample(1, preds[:,:-1].flatten())
                token = index_dict[int(pw)]
        initial.append(token)
        inputs = np.c_[inputs, pw]
        count += 1
        if count == num:
            done = True
    return initial[net.context:]


def compute_bleu(net, word_dict, index_dict, tokens, initial=None, IM=None):
    """
    Return BLEU scores for reference tokens
    For each reference caption, a candidate caption is sampled from net
    """
    bleu_scores = np.zeros((len(tokens), 3))
    for i, ref in enumerate(tokens):
        if initial != None:
            init = copy.deepcopy(initial)
        else:
            init = None
        ref = ref[net.context:][:-1]
        if IM != None:
            can = sample(net, word_dict, index_dict, len(ref), IM[i], initial=init)
        else:
            can = sample(net, word_dict, index_dict, len(ref), initial=init)

        # Compute bleu using n = (1,2,3)
        n1 = bleu.score_cooked([bleu.cook_test(can, bleu.cook_refs([ref], n=1), n=1)], n=1)
        n2 = bleu.score_cooked([bleu.cook_test(can, bleu.cook_refs([ref], n=2), n=2)], n=2)
        n3 = bleu.score_cooked([bleu.cook_test(can, bleu.cook_refs([ref], n=3), n=3)], n=3)
        bleu_scores[i] = [n1,n2,n3]

    return bleu_scores


def nn(net, im, IM, k=1):
    """
    Return the k-NN of im to images in IM
    """
    d = np.zeros(len(IM))
    for i in range(len(IM)):
        d[i] = norm(im - IM[i])
    inds = np.argsort(d)[:k]
    return inds


def imquery(net, Im, tokens, word_dict, IM, mean_im, k=1):
    """
    Return the k-NN captions that describe Im
    """
    scores = np.zeros(len(tokens))
    ngrams = get_ngrams(tokens, context=net.context)
    for i, x in enumerate(ngrams):
        num = perplexity(net, [x], word_dict, Im=Im, context=net.context)
        denom = perplexity(net, [x], word_dict, Im=mean_im, context=net.context)
        scores[i] = num / denom
    inds = np.argsort(scores)[:k]
    return [tokens[i][net.context:][:-1] for i in inds]


def txt2im(net, txt, IM, word_dict, k):
    """
    Return the nearest indices of images in IM to txt
    """
    scores = np.zeros(len(IM))
    if txt[0] != '<start>':
        txt = ['<start>'] * net.context + txt
    ngrams = get_ngrams([txt], context=net.context)
    for i in range(len(IM)):
        scores[i] = perplexity(net, ngrams, word_dict, [IM[i]], context=net.context)
    inds = np.argsort(scores)[:k]
    return inds
    

def im2txt(net, Im, word_dict, X, IM, k=5, shortlist=15):
    """
    Retrieve text given images
    """
    # Compute the k-NN
    nearest = nn(net, Im, IM, shortlist)
    X = [X[i] for i in nearest]
    IM_near = IM[nearest]
    mean_im = IM.mean(0).reshape(1, IM.shape[1])   
    captions = imquery(net, [Im], X, word_dict, IM, [mean_im], k)
    return captions




