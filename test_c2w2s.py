'''
Tweet2Vec classifier trainer
'''

import numpy as np
import lasagne
import theano
import theano.tensor as T
import random
import sys
import batch
import time
import cPickle as pkl
import io

from collections import OrderedDict
from t2v import char2word2vec, load_params
from settings import NUM_EPOCHS, N_BATCH, MAX_LENGTH, N_CHAR, SCALE, SDIM, MAX_CLASSES, LEARNING_RATE, DISPF, SAVEF, REGULARIZATION, RELOAD_DATA, RELOAD_MODEL

def classify(tweet, t_mask, params, n_classes, n_chars):
    # tweet embedding
    emb_layer = char2word2vec(tweet, t_mask, params, n_chars)

    # Dense layer for classes
    l_dense = lasagne.layers.DenseLayer(emb_layer[2], n_classes, W=params['W_cl'], b=params['b_cl'], nonlinearity=lasagne.nonlinearities.softmax)

    return lasagne.layers.get_output(l_dense), emb_layer[0]

def main(args):

    data_path = args[0]
    model_path = args[1]
    save_path = args[2]
    if len(args)>3:
        m_num = int(args[3])

    print("Preparing Data...")

    # Test data
    Xt = []
    yt = []
    with io.open(data_path,'r',encoding='utf-8') as f:
        for line in f:
            (yc, Xc) = line.rstrip('\n').split('\t')
            Xt.append(Xc[:MAX_LENGTH])
            yt.append(yc.split(','))

    # Model
    print("Loading model params...")
    if len(args)>3:
        params = load_params('%s/model_%d.npz' % (model_path,m_num))
    else:
        params = load_params('%s/model.npz' % model_path)

    print("Loading dictionaries...")
    with open('%s/dict.pkl' % model_path, 'rb') as f:
        chardict = pkl.load(f)
    with open('%s/label_dict.pkl' % model_path, 'rb') as f:
        labeldict = pkl.load(f)
    n_char = len(chardict.keys()) + 1
    n_classes = min(len(labeldict.keys()) + 1, MAX_CLASSES)

    # iterators
    test_iter = batch.BatchTweets(Xt, yt, labeldict, batch_size=N_BATCH, max_classes=MAX_CLASSES, test=True)

    print("Building network...")

    # Tweet variables
    tweet = T.itensor4()
    targets = T.imatrix()

    # masks
    t_mask = T.ftensor3()

    # network for prediction
    predictions = classify(tweet, t_mask, params, n_classes, n_char)[0]
    embeddings = classify(tweet, t_mask, params, n_classes, n_char)[1]

    # Theano function
    print("Compiling theano functions...")
    predict = theano.function([tweet,t_mask],predictions)
    encode = theano.function([tweet,t_mask],embeddings)

    # Test
    print("Testing...")
    out_data = []
    out_pred = []
    out_emb = []
    out_target = []
    for xr,y in test_iter:
        x, x_m = batch.prepare_data_c2w2s(xr, chardict, n_chars=n_char)
        p = predict(x,x_m)
        e = encode(x,x_m)

        for idx, item in enumerate(xr):
            out_data.append(item)
            out_pred.append(p[idx,:])
            out_emb.append(e[idx,:])
            out_target.append(y[idx])

    # Save
    print("Saving...")
    with open('%s/data.pkl'%save_path,'w') as f:
        pkl.dump(out_data,f)
    with open('%s/predictions.pkl'%save_path,'w') as f:
        pkl.dump(out_pred,f)
    with open('%s/embeddings.pkl'%save_path,'w') as f:
        pkl.dump(out_emb,f)
    with open('%s/targets.pkl'%save_path,'w') as f:
        pkl.dump(out_target,f)


if __name__ == '__main__':
    main(sys.argv[1:])
