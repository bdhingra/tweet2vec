import numpy as np
import cPickle as pkl
import codecs

from collections import OrderedDict
from settings_word import MAX_LENGTH, WORD_LEVEL

class BatchTweets():

    def __init__(self, data, targets, labeldict, batch_size=128, max_classes=1000, test=False):
        # convert targets to indices
        if not test:
            tags = []
            for l in targets:
                tags.append(labeldict[l] if l in labeldict and labeldict[l] < max_classes else 0)
        else:
            tags = []
            for line in targets:
                tags.append([labeldict[l] if l in labeldict and labeldict[l] < max_classes else 0 for l in line])

        self.batch_size = batch_size
        self.data = data
        self.targets = tags

        self.prepare()
        self.reset()

    def prepare(self):
        self.indices = np.arange(len(self.data))
        self.curr_indices = np.random.permutation(self.indices)

    def reset(self):
        self.curr_indices = np.random.permutation(self.indices)
        self.curr_pos = 0
        self.curr_remaining = len(self.curr_indices)

    def next(self):
        if self.curr_pos >= len(self.indices):
            self.reset()
            raise StopIteration()

        # current batch size
        curr_batch_size = np.minimum(self.batch_size, self.curr_remaining)

        # indices for current batch
        curr_indices = self.curr_indices[self.curr_pos:self.curr_pos+curr_batch_size]
        self.curr_pos += curr_batch_size
        self.curr_remaining -= curr_batch_size

        # data and targets for current batch
        x = [self.data[ii] for ii in curr_indices]
        y = [self.targets[ii] for ii in curr_indices]

        return x, y

    def __iter__(self):
        return self

def prepare_data(seqs_x, tokendict, n_tokens=1000):
    """
    Prepare the data for training - add masks and remove infrequent tokens
    """
    seqsX = []
    for cc in seqs_x:
        if (WORD_LEVEL):
            seqsX.append([tokendict[c] if c in tokendict and tokendict[c] < n_tokens else 0 for c in cc.split()[:MAX_LENGTH]])
        else:
            seqsX.append([tokendict[c] if c in tokendict and tokendict[c] < n_tokens else 0 for c in list(cc)[:MAX_LENGTH]])
    seqs_x = seqsX

    lengths_x = [len(s) for s in seqs_x]

    n_samples = len(seqs_x)

    x = np.zeros((n_samples,MAX_LENGTH)).astype('int32')
    x_mask = np.zeros((n_samples,MAX_LENGTH)).astype('float32')
    for idx, s_x in enumerate(seqs_x):
        x[idx,:lengths_x[idx]] = s_x
        x_mask[idx,:lengths_x[idx]] = 1.

    return np.expand_dims(x, axis=2), x_mask

def build_dictionary(text):
    """
    Build a dictionary of characters or words
    text: list of tweets
    """
    tokencount = OrderedDict()

    for cc in text:       
        if WORD_LEVEL:
            tokens = cc.split()
        else:
            tokens = list(cc)
        for c in tokens:
            if c not in tokencount:
                tokencount[c] = 0
            tokencount[c] += 1

    tokens = tokencount.keys()
    freqs = tokencount.values()
    sorted_idx = np.argsort(freqs)[::-1]

    tokendict = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):
        tokendict[tokens[sidx]] = idx + 1

    return tokendict, tokencount

def save_dictionary(worddict, wordcount, loc):
    """
    Save a dictionary to the specified location 
    """
    with open(loc, 'w') as f:
        pkl.dump(worddict, f)
        pkl.dump(wordcount, f)

def build_label_dictionary(targets):
    """
    Build a label dictionary
    targets: list of labels, each item may have multiple labels
    """
    labelcount = OrderedDict()
    for l in targets:
        if l not in labelcount:
            labelcount[l] = 0
        labelcount[l] += 1
    labels = labelcount.keys()
    freqs = labelcount.values()
    sorted_idx = np.argsort(freqs)[::-1]

    labeldict = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):
        labeldict[labels[sidx]] = idx + 1

    return labeldict, labelcount
