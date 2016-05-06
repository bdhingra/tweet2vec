'''
Tweet2Vec classifier trainer
'''

import numpy as np
import lasagne
import theano
import theano.tensor as T
import random
import sys
import batch_word as batch
import time
import cPickle as pkl
import io
import shutil

from collections import OrderedDict
from w2v import tweet2vec, init_params, load_params_shared
from settings_word import NUM_EPOCHS, N_BATCH, N_WORD, SCALE, WDIM, MAX_CLASSES, LEARNING_RATE, DISPF, SAVEF, REGULARIZATION, RELOAD_MODEL, MOMENTUM, SCHEDULE
from evaluate import precision

T1 = 0.01
T2 = 0.0001

def schedule(lr, mu):
    print("Updating Schedule...")
    lr = max(1e-5,lr/2)
    return lr, mu

def tnorm(tens):
    '''
    Tensor Norm
    '''
    return T.sqrt(T.sum(T.sqr(tens),axis=1))

def classify(tweet, t_mask, params, n_classes, n_tokens):
    # tweet embedding
    emb_layer = tweet2vec(tweet, t_mask, params, n_tokens)

    # Dense layer for classes
    l_dense = lasagne.layers.DenseLayer(emb_layer, n_classes, W=params['W_cl'], b=params['b_cl'], nonlinearity=lasagne.nonlinearities.softmax)

    return lasagne.layers.get_output(l_dense), l_dense, lasagne.layers.get_output(emb_layer)

def main(train_path,val_path,save_path,num_epochs=NUM_EPOCHS):
    global T1

    # save settings
    shutil.copyfile('settings_word.py','%s/settings_word.txt'%save_path)

    print("Preparing Data...")
    # Training data
    Xt = []
    yt = []
    with io.open(train_path,'r',encoding='utf-8') as f:
        for line in f:
            (yc, Xc) = line.rstrip('\n').split('\t')
            Xt.append(Xc)
            yt.append(yc)
    # Validation data
    Xv = []
    yv = []
    with io.open(val_path,'r',encoding='utf-8') as f:
        for line in f:
            (yc, Xc) = line.rstrip('\n').split('\t')
            Xv.append(Xc)
            yv.append(yc.split(','))

    print("Preparing Model...")
    if not RELOAD_MODEL:
        # Build dictionaries from training data
        tokendict, tokencount = batch.build_dictionary(Xt)
        n_token = min(len(tokendict.keys()) + 1, N_WORD)
        batch.save_dictionary(tokendict,tokencount,'%s/dict.pkl' % save_path)
        # params
        params = init_params(n_chars=n_token)
        
        labeldict, labelcount = batch.build_label_dictionary(yt)
        batch.save_dictionary(labeldict, labelcount, '%s/label_dict.pkl' % save_path)

        n_classes = min(len(labeldict.keys()) + 1, MAX_CLASSES)

        # classification params
        params['W_cl'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(WDIM,n_classes)).astype('float32'), name='W_cl')
        params['b_cl'] = theano.shared(np.zeros((n_classes)).astype('float32'), name='b_cl')

    else:
        print("Loading model params...")
        params = load_params_shared('%s/best_model.npz' % save_path)

        print("Loading dictionaries...")
        with open('%s/dict.pkl' % save_path, 'rb') as f:
            tokendict = pkl.load(f)
        with open('%s/label_dict.pkl' % save_path, 'rb') as f:
            labeldict = pkl.load(f)
        n_token = min(len(tokendict.keys()) + 1, N_WORD)
        n_classes = min(len(labeldict.keys()) + 1, MAX_CLASSES)

    # iterators
    train_iter = batch.BatchTweets(Xt, yt, labeldict, batch_size=N_BATCH, max_classes=MAX_CLASSES)
    val_iter = batch.BatchTweets(Xv, yv, labeldict, batch_size=N_BATCH, max_classes=MAX_CLASSES, test=True)

    print("Building network...")
    # Tweet variables
    tweet = T.itensor3()
    targets = T.ivector()

    # masks
    t_mask = T.fmatrix()

    # network for prediction
    predictions, net, emb = classify(tweet, t_mask, params, n_classes, n_token)

    # batch loss
    loss = lasagne.objectives.categorical_crossentropy(predictions, targets)
    cost = T.mean(loss) + REGULARIZATION*lasagne.regularization.regularize_network_params(net, lasagne.regularization.l2)
    cost_only = T.mean(loss)
    reg_only = REGULARIZATION*lasagne.regularization.regularize_network_params(net, lasagne.regularization.l2)

    # params and updates
    print("Computing updates...")
    lr = LEARNING_RATE
    mu = MOMENTUM
    updates = lasagne.updates.nesterov_momentum(cost, lasagne.layers.get_all_params(net), lr, momentum=mu)

    # Theano function
    print("Compiling theano functions...")
    inps = [tweet,t_mask,targets]
    predict = theano.function([tweet,t_mask],predictions)
    encode = theano.function([tweet,t_mask],emb)
    cost_val = theano.function(inps,[cost_only,emb])
    train = theano.function(inps,cost,updates=updates)
    reg_val = theano.function([],reg_only)

    # Training
    print("Training...")
    uidx = 0
    maxp = 0.
    start = time.time()
    valcosts = []
    try:
	for epoch in range(num_epochs):
	    n_samples = 0
            train_cost = 0.
	    print("Epoch {}".format(epoch))

            # learning schedule
            if len(valcosts) > 1 and SCHEDULE:
                change = (valcosts[-1]-valcosts[-2])/abs(valcosts[-2])
                if change < T1:
                    lr, mu = schedule(lr, mu)
                    updates = lasagne.updates.nesterov_momentum(cost, lasagne.layers.get_all_params(net), lr, momentum=mu)
                    train = theano.function(inps,cost,updates=updates)
                    T1 = T1/2

            # stopping criterion
            if len(valcosts) > 6:
                deltas = []
                for i in range(5):
                    deltas.append((valcosts[-i-1]-valcosts[-i-2])/abs(valcosts[-i-2]))
                if sum(deltas)/len(deltas) < T2:
                    break

            ud_start = time.time()
	    for xr,y in train_iter:
		n_samples +=len(xr)
		uidx += 1
		x, x_m = batch.prepare_data(xr, tokendict, n_tokens=n_token)
		if x==None:
		    print("Minibatch with zero samples under maxlength.")
		    uidx -= 1
		    continue

		curr_cost = train(x,x_m,y)
                train_cost += curr_cost*len(xr)
		ud = time.time() - ud_start

		if np.isnan(curr_cost) or np.isinf(curr_cost):
		    print("Nan detected.")
		    return

		if np.mod(uidx, DISPF) == 0:
		    print("Epoch {} Update {} Cost {} Time {}".format(epoch,uidx,curr_cost,ud))

		if np.mod(uidx,SAVEF) == 0:
		    print("Saving...")
		    saveparams = OrderedDict()
		    for kk,vv in params.iteritems():
			saveparams[kk] = vv.get_value()
		    np.savez('%s/model.npz' % save_path,**saveparams)
		    print("Done.")

            print("Testing on Validation set...")
            preds = []
            targs = []
	    for xr,y in val_iter:
		x, x_m = batch.prepare_data(xr, tokendict, n_tokens=n_token)
		if x==None:
                    print("Validation: Minibatch with zero samples under maxlength.")
		    continue

                vp = predict(x,x_m)
                ranks = np.argsort(vp)[:,::-1]
                for idx,item in enumerate(xr):
                    preds.append(ranks[idx,:])
                    targs.append(y[idx])

            validation_cost = precision(np.asarray(preds),targs,1)
            regularization_cost = reg_val()

            if validation_cost > maxp:
                maxp = validation_cost
                saveparams = OrderedDict()
                for kk,vv in params.iteritems():
                    saveparams[kk] = vv.get_value()
                np.savez('%s/best_model.npz' % (save_path),**saveparams)

	    print("Epoch {} Training Cost {} Validation Precision {} Regularization Cost {} Max Precision {}".format(epoch, train_cost/n_samples, validation_cost, regularization_cost, maxp))
	    print("Seen {} samples.".format(n_samples))
            valcosts.append(validation_cost)

            print("Saving...")
            saveparams = OrderedDict()
            for kk,vv in params.iteritems():
                saveparams[kk] = vv.get_value()
            np.savez('%s/model_%d.npz' % (save_path,epoch),**saveparams)
            print("Done.")

    except KeyboardInterrupt:
	pass
    print("Total training time = {}".format(time.time()-start))

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2],sys.argv[3])
