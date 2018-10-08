import operator
from collections import defaultdict, Counter
from gtnlplib.constants import START_TAG,END_TAG, UNK
import numpy as np
import torch
import torch.nn
from torch import autograd
from torch.autograd import Variable

def get_torch_variable(arr):
    # returns a pytorch variable of the array
    torch_var = torch.autograd.Variable(torch.from_numpy(np.array(arr).astype(np.float32)))
    return torch_var.view(1,-1)

def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def viterbi_step(all_tags, tag_to_ix, cur_tag_scores, transition_scores, prev_scores):
    """
    Calculates the best path score and corresponding back pointer for each tag for a word in the sentence in pytorch, which you will call from the main viterbi routine.
    
    parameters:
    - all_tags: list of all tags: includes both the START_TAG and END_TAG
    - tag_to_ix: a dictionary that maps each tag (including the START_TAG and the END_TAG) to a unique index.
    - cur_tag_scores: pytorch Variable that contains the local emission score for each tag for the current token in the sentence
                       it's size is : [ len(all_tags) ] 
    - transition_scores: pytorch Variable that contains the tag_transition_scores
                        it's size is : [ len(all_tags) x len(all_tags) ] 
    - prev_scores: pytorch Variable that contains the scores for each tag for the previous token in the sentence: 
                    it's size is : [ 1 x len(all_tags) ] 
    
    :returns:
    - viterbivars: a list of pytorch Variables such that each element contains the score for each tag in all_tags for the current token in the sentence
    - bptrs: a list of idx that contains the best_previous_tag for each tag in all_tags for the current token in the sentence
    """
    viterbivars = []
    bptrs = []
    # make sure end_tag exists in all_tags
    for next_tag in list(all_tags):
        cur_tag_idx = tag_to_ix[next_tag]
        emission_score = cur_tag_scores[cur_tag_idx]
        tag_trans_score = transition_scores[cur_tag_idx].view(1, -1)
        product = tag_trans_score + prev_scores + emission_score
        best_score, best_prev_tag = torch.max(product, 1)
        best_prev_tag = best_prev_tag.data.numpy()[0]
        viterbivars.insert(cur_tag_idx, best_score)
        bptrs.insert(cur_tag_idx, best_prev_tag)

    return viterbivars, bptrs


def build_trellis(all_tags, tag_to_ix, cur_tag_scores, transition_scores):
    """
    This function should compute the best_path and the path_score. 
    Use viterbi_step to implement build_trellis in viterbi.py in Pytorch.
    
    parameters:
    - all_tags: a list of all tags: includes START_TAG and END_TAG
    - tag_to_ix: a dictionary that maps each tag to a unique id.
    - cur_tag_scores: a list of pytorch Variables where each contains the local emission score for each tag for that particular token in the sentence, len(cur_tag_scores) will be equal to len(words)
                        it's size is : [ len(words in sequence) x len(all_tags) ] 
    - transition_scores: pytorch Variable (a matrix) that contains the tag_transition_scores
                        it's size is : [ len(all_tags) x len(all_tags) ] 
    
    :returns:
    - path_score: the score for the best_path
    - best_path: the actual best_path, which is the list of tags for each token: exclude the START_TAG and END_TAG here.
    """
    
    ix_to_tag={ v:k for k,v in tag_to_ix.items() }
    
    # make sure END_TAG is in all_tags
    # if (END_TAG not in all_tags):
    #     all_tags.append(END_TAG)

    # if (START_TAG not in all_tags):
    #     [START_TAG] + all_tags

    # setting all the initial score to START_TAG
    initial_vec = np.full((1,len(all_tags)),-np.inf)
    initial_vec[0][tag_to_ix[START_TAG]] = 0
    prev_scores = torch.autograd.Variable(torch.from_numpy(initial_vec.astype(np.float32))).view(1,-1)
    whole_bptrs=[]
    scores = []
    # get the best path for 
    for m in range(len(cur_tag_scores)):
        emission_score = cur_tag_scores[m]
        score, backpointer = viterbi_step(all_tags, tag_to_ix, emission_score, transition_scores, prev_scores)
        score_arr = np.full((1,len(all_tags)),-np.inf)
        for i in range(len(score)):
            score_arr[0][i] = score[i].data.numpy()
        score_var = torch.autograd.Variable(torch.from_numpy(score_arr.astype(np.float32))).view(1,-1)
        prev_scores = score_var
        whole_bptrs.insert(m, backpointer)
        scores.insert(m, score_arr)  

    initial_vec = np.full((1, len(all_tags)), -np.inf)
    initial_vec[0][tag_to_ix[END_TAG]] = 0
    end_tag_emission_score = torch.autograd.Variable(torch.from_numpy(initial_vec.astype(np.float32))).view(-1, 1)
    end_tag_score, end_tag_backpointer = viterbi_step(all_tags, tag_to_ix, end_tag_emission_score, transition_scores, prev_scores)
    score = np.array(max([x.data.numpy() for x in end_tag_score]))
    score_arr = np.array([x.data.numpy() for x in end_tag_score])
    path_score = torch.autograd.Variable(torch.from_numpy(score.astype(np.float32)))
    best_path = [all_tags[end_tag_backpointer[score_arr.argmax()]]] # tag for END

    for i in range(len(whole_bptrs) - 1, 0, -1):
        backpointer = whole_bptrs[i]
        best_path.append(all_tags[backpointer[scores[i].argmax()]])

    #best_path.append(all_tags[end_tag_backpointer[score_arr.argmax()]])

    return path_score, best_path[::-1]

    
