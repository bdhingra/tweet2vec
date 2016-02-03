import sys
import numpy as np
import cPickle as pkl
import io

seedvec = np.asarray(pkl.load(open(sys.argv[1],'r')))
tweetvec = np.asarray(pkl.load(open(sys.argv[2]+'embeddings.pkl','r')))
tweets = pkl.load(open(sys.argv[2]+'data.pkl','r'))

proto = np.mean(seedvec,axis=0)
proto = proto/np.linalg.norm(proto)

tweetvec_norm = np.sum(np.abs(tweetvec)**2,axis=1)**(1./2)
tweetvec = tweetvec/tweetvec_norm[:,None]

scores = np.sum(np.tile(proto,(tweetvec.shape[0],1))*tweetvec,axis=1)
index = np.argsort(scores)[::-1]

with io.open(sys.argv[3],'w') as f:
    for idx in index:
        f.write(tweets[idx]+'\n')
