from gtnlplib.constants import OFFSET
from gtnlplib import clf_base, evaluation, preproc

import math
import numpy as np
from collections import defaultdict

def get_nb_weights(trainfile, smoothing):
    """
    estimate_nb function assumes that the labels are one for each document, where as in POS tagging: we have labels for 
    each particular token. So, in order to calculate the emission score weights: P(w|y) for a particular word and a 
    token, we slightly modify the input such that we consider each token and its tag to be a document and a label. 
    The following helper code converts the dataset to token level bag-of-words feature vector and labels. 
    The weights obtained from here will be used later as emission scores for the viterbi tagger.
    
    inputs: train_file: input file to obtain the nb_weights from
    smoothing: value of smoothing for the naive_bayes weights
    
    :returns: nb_weights: naive bayes weights
    """
    token_level_docs=[]
    token_level_tags=[]
    for words,tags in preproc.conll_seq_generator(trainfile):
        token_level_docs += [{word:1} for word in words]
        token_level_tags +=tags
    nb_weights = estimate_nb(token_level_docs, token_level_tags, smoothing)
    
    return nb_weights


def get_corpus_counts(x,y,label):
    """
    Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    corpus_counts = defaultdict(float)
    for i in range(len(y)):
        if y[i] == label:
            counter = x[i]
            for word in counter:
                corpus_counts[word] += counter[word]

    return corpus_counts


def estimate_pxy(x,y,label,smoothing,vocab):
    '''
    Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    '''

    log_probabilities = defaultdict(float)
    corpus_counts = get_corpus_counts(x, y, label)
    total = sum(corpus_counts.values())
    for word in vocab:
        log_probabilities[word] = np.log(((corpus_counts[word] if word in corpus_counts else 0) + smoothing) / (total + len(vocab) * smoothing))
    return log_probabilities


def estimate_nb(x,y,smoothing):
    """
    estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """

    labels = set(y)
    counts = defaultdict(float)
    doc_counts = defaultdict(float)

    num_instances = len(x)

    for genre in y:
        doc_counts[genre] += 1

    words = set()
    for counter in x:
        words.update(list(counter.keys()))

    vocab = list(words)

    for label in labels:
        pxy = estimate_pxy(x, y, label, smoothing, vocab)
        for word in pxy:
            counts[(label, word)] = pxy[word]

    for genre in doc_counts:
        counts[(genre, OFFSET)] = math.log(doc_counts[genre] / num_instances)

    return counts


def find_best_smoother(x_tr,y_tr,x_dv,y_dv,smoothers):
    '''
    find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values
    :returns: best smoothing value
    :rtype: float

    '''

    accuracy = {}
    genres = set(y_dv)
    for smoother in smoothers:
        accuracy[smoother] = evaluation.acc(clf_base.predict_all(x_dv,
            estimate_nb(x_tr, y_tr, smoother), genres), y_dv)

    best_smoother = clf_base.argmax(accuracy)
    return best_smoother, accuracy
    







