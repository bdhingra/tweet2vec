import numpy as np
import cPickle as pkl
import codecs

from collections import OrderedDict
from settings_char import MAX_LENGTH

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

def prepare_data(seqs_x, chardict, n_chars=1000):
    """
    Prepare the data for training - add masks and remove infrequent characters
    """
    seqsX = []
    for cc in seqs_x:
        seqsX.append([chardict[c] if c in chardict and chardict[c] <= n_chars else 0 for c in list(cc)])
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
    Build a character dictionary
    text: list of tweets
    """
    charcount = OrderedDict()
    for cc in text:
        chars = list(cc)
        for c in chars:
            if c not in charcount:
                charcount[c] = 0
            charcount[c] += 1
    chars = charcount.keys()
    freqs = charcount.values()
    sorted_idx = np.argsort(freqs)[::-1]

    chardict = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):
        chardict[chars[sidx]] = idx + 1

    return chardict, charcount

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

'''
Obsolete
'''
def prepare_data_c2w2s(seqs_x, chardict, n_chars=1000):
    """
    Put the data into format useable by the model
    """
    n_samples = len(seqs_x)
    x = np.zeros((n_samples,MAX_SEQ_LENGTH,MAX_WORD_LENGTH)).astype('int32')
    x_mask = np.zeros((n_samples,MAX_SEQ_LENGTH,MAX_WORD_LENGTH)).astype('float32')

    # Split words and replace by indices
    for seq_id, cc in enumerate(seqs_x):
        words = cc.split()
        for word_id, word in enumerate(words):
            if word_id >= MAX_SEQ_LENGTH:
                break
            c_len = min(MAX_WORD_LENGTH, len(word))
            x[seq_id,word_id,:c_len] = [chardict[c] if c in chardict and chardict[c] < n_chars else 0 for c in list(word)[:c_len]]
            x_mask[seq_id,word_id,:c_len] = 1.

    return np.expand_dims(x,axis=3), x_mask

