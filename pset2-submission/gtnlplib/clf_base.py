from gtnlplib.constants import OFFSET

import operator

# use this to find the highest-scoring label
argmax = lambda x : max(x.items(),key=operator.itemgetter(1))[0]

def make_feature_vector(base_features,label):
    """take a counter of base features and a label; return a dict of features, corresponding to f(x,y)

    :param base_features: counter of base features
    :param label: label string
    :returns: dict of features, f(x,y)
    :rtype: dict

    """
    feature_dict = {}
    for feature in base_features:
        feature_dict[(label, feature)] = base_features[feature]

    feature_dict[(label, OFFSET)] = 1
    return feature_dict
    

def predict(base_features,weights,labels):
    """prediction function

    :param base_features: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,base_feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict

    """
    scores = {}
    for label in labels:
        scores[label] = 0.0
        for feature in base_features:
            scores[label] += base_features[feature] * weights[(label, feature)]
        scores[label] += weights[(label, OFFSET)]

    return argmax(scores), scores
    
