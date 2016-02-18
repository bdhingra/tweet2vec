import sys
import numpy as np
import io
import cPickle as pkl

MAX_LENGTH = 20
N_CHAR = 30000

datafile = sys.argv[1]
dictfile = sys.argv[2]
N = int(sys.argv[3])
outfile = sys.argv[4]
 
print("Preparing Data...")
Xt = []
yt = []
with io.open(datafile,'r',encoding='utf-8') as f:
    for line in f:
        (yc, Xc) = line.rstrip('\n').split('\t')
        Xt.append(Xc)
        yt.append(yc.split(','))

print("Loading dictionary...")
with open(dictfile, 'rb') as f:
    chardict = pkl.load(f)

print("Counting rare words...")
rare_counts = np.zeros((len(Xt)))
for idx, seq in enumerate(Xt):
    for w in seq.split()[:MAX_LENGTH]:
        if w not in chardict or chardict[w] >= N_CHAR:
            rare_counts[idx] += 1
    rare_counts[idx] /= min(len(seq.split()),MAX_LENGTH)

print("Filtering tweets...")
sorted_index = np.argsort(rare_counts)[::-1][:N]
#sorted_index = np.argsort(rare_counts)[:N]
rare_tweets = [Xt[ii] for ii in sorted_index]
rare_labels = [yt[ii] for ii in sorted_index]

print("Saving...")
with io.open(outfile,'w',encoding='utf-8') as f:
    for idx,t in enumerate(rare_tweets):
        f.write('%s\t%s\n'%(','.join(rare_labels[idx]),t))
