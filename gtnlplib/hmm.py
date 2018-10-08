from gtnlplib.preproc import conll_seq_generator
from gtnlplib.constants import START_TAG, END_TAG, OFFSET, UNK
from gtnlplib import naive_bayes, most_common 
import numpy as np
from collections import defaultdict
import torch
import torch.nn
from torch.autograd import Variable


def compute_transition_weights(trans_counts, smoothing):
    """
    Compute the HMM transition weights, given the counts.
    Don't forget to assign smoothed probabilities to transitions which
    do not appear in the counts.
    
    This will also affect your computation of the denominator.
    Don't forget to assign smoothed probabilities to transitions which do not appear in the counts.
Do not assign probabilities for transitions to the START_TAG, which can only come first. This will also affect your computation of the denominator, since you are not smoothing the probability of transitions to the START_TAG.
Don't forget to assign probabilities to transitions to the END_TAG; this too will affect your denominator.
As always, probabilities should sum to one (this time conditioned on the previous tag)

    :param trans_counts: counts, generated from most_common.get_tag_trans_counts
    :param smoothing: additive smoothing
    :returns: dict of features [(curr_tag,prev_tag)] and weights

    """
    weights = defaultdict(float)
    
    total_count = {}
    for tag in trans_counts.keys():
        total_count[tag] = sum(trans_counts[tag].values())
        

    for prev_tag in trans_counts:
        for curr_tag in (list(trans_counts.keys()) + [END_TAG]):
            if curr_tag in trans_counts[prev_tag]:
                weights[(curr_tag, prev_tag)] = np.log((trans_counts[prev_tag][curr_tag] + smoothing) / (total_count[prev_tag] + len(trans_counts) * smoothing))
            else:
                weights[(curr_tag, prev_tag)] = np.log(smoothing / (total_count[prev_tag] + len(trans_counts) * smoothing))


    for tag in (list(trans_counts.keys()) + [END_TAG]):
        weights[START_TAG, tag] = -np.inf
        weights[tag, END_TAG] = -np.inf

    return weights


def compute_weights_variables(nb_weights, hmm_trans_weights, vocab, word_to_ix, tag_to_ix):
    """
    Computes autograd Variables of two weights: emission_probabilities and the tag_transition_probabilties
    parameters:
    nb_weights: -- a dictionary of emission weights
    hmm_trans_weights: -- dictionary of tag transition weights
    vocab: -- list of all the words
    word_to_ix: -- a dictionary that maps each word in the vocab to a unique index
    tag_to_ix: -- a dictionary that maps each tag (including the START_TAG and the END_TAG) to a unique index.
    
    :returns:
    emission_probs_vr: torch Variable of a matrix of size Vocab x Tagset_size
    tag_transition_probs_vr: torch Variable of a matrix of size Tagset_size x Tagset_size
    :rtype: autograd Variables of the the weights
    """
    # Assume that tag_to_ix includes both START_TAG and END_TAG
    
    tag_transition_probs = np.full((len(tag_to_ix), len(tag_to_ix)), -np.inf)
    emission_probs = np.full((len(vocab),len(tag_to_ix)), -np.inf)

    for word in word_to_ix.keys():
        for tag in tag_to_ix.keys():
            if (tag, word) in nb_weights and word != OFFSET:
                weight = nb_weights[tag, word]
                emission_probs[word_to_ix[word], tag_to_ix[tag]] = weight
            elif tag != START_TAG and tag != END_TAG:
                emission_probs[word_to_ix[word], tag_to_ix[tag]] = 0


    for tag, prev_tag in hmm_trans_weights:
        if prev_tag != END_TAG or tag != START_TAG:
            weight = hmm_trans_weights[tag, prev_tag]
            tag_transition_probs[tag_to_ix[tag], tag_to_ix[prev_tag]] = weight

    
    emission_probs_vr = Variable(torch.from_numpy(emission_probs.astype(np.float32)))
    tag_transition_probs_vr = Variable(torch.from_numpy(tag_transition_probs.astype(np.float32)))
    
    return emission_probs_vr, tag_transition_probs_vr
    
