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
import shutil

from collections import OrderedDict
from t2v import char2word2vec, init_params_c2w2s, load_params_shared
from settings import NUM_EPOCHS, N_BATCH, MAX_LENGTH, N_CHAR, SCALE, SDIM, MAX_CLASSES, LEARNING_RATE, DISPF, SAVEF, REGULARIZATION, RELOAD_DATA, RELOAD_MODEL, DEBUG, MOMENTUM, TRANSFER

def tnorm(tens):
    '''
    Tensor Norm
    '''
    return T.sqrt(T.sum(T.sqr(tens),axis=1))

def classify(tweet, t_mask, params, n_classes, n_chars):
    # tweet embedding
    emb_layer = char2word2vec(tweet, t_mask, params, n_chars)

    # Dense layer for classes
    l_dense = lasagne.layers.DenseLayer(emb_layer[2], n_classes, W=params['W_cl'], b=params['b_cl'], nonlinearity=lasagne.nonlinearities.softmax, name='linear-softmax')

    return lasagne.layers.get_output(l_dense), l_dense, emb_layer[1], emb_layer[2]

def print_params(params):
    for kk,vv in params.iteritems():
        print("Param {} = {}".format(kk, vv.get_value()))

def display_actv(x, x_m, y, tweet, t_mask, targets, net, prefix):
    print("\nactivations...")

    layers = lasagne.layers.get_all_layers(net)

    inps = [tweet,t_mask,targets]

    for l in layers:
        f = theano.function(inps, lasagne.layers.get_output(l),on_unused_input='warn')
        print("layer "+prefix+" {} - {}".format(l.name, f(x,x_m,y)))

def main(train_path,val_path,save_path,num_epochs=NUM_EPOCHS):

    # save settings
    shutil.copyfile('settings.py','%s/settings.txt'%save_path)

    print("Preparing Data...")

    if not RELOAD_DATA:
        # Training data
        Xt = []
        yt = []
        with io.open(train_path,'r',encoding='utf-8') as f:
            for idx,line in enumerate(f):
                row = line.rstrip('\n').split('\t')
                if len(row) != 2:
                    print("Something wrong on line {}".format(idx))
                    continue
                (yc, Xc) = row
                Xt.append(Xc[:MAX_LENGTH])
                yt.append(yc)

        # Validation data
        Xv = []
        yv = []
        with io.open(val_path,'r',encoding='utf-8') as f:
            for line in f:
                row = line.rstrip('\n').split('\t')
                if len(row) != 2:
                    print("Something wrong on line {}".format(idx))
                    continue
                (yc, Xc) = row
                Xv.append(Xc[:MAX_LENGTH])
                yv.append(yc)

        with open('%s/train_data.pkl'%(save_path),'w') as f:
            pkl.dump(Xt, f)
            pkl.dump(yt, f)
        with open('%s/val_data.pkl'%(save_path),'w') as f:
            pkl.dump(Xv, f)
            pkl.dump(yv, f)
    else:
        print("Loading Data...")
        with open(train_path,'r') as f:
            Xt = pkl.load(f)
            yt = pkl.load(f)
        with open(val_path,'r') as f:
            Xv = pkl.load(f)
            yv = pkl.load(f)

    if not RELOAD_MODEL:
        # transfer
        if TRANSFER != None:
            params = load_params_shared(TRANSFER)
            with open('%s/dict.pkl' % TRANSFER.rsplit('/',1)[0], 'rb') as f:
                chardict = pkl.load(f)
            n_char = len(chardict.keys()) + 1
            shutil.copyfile('%s/dict.pkl' % TRANSFER.rsplit('/',1)[0], '%s/dict.pkl' % save_path)
        else:
            # Build dictionaries from training data
            chardict, charcount = batch.build_dictionary(Xt)
            n_char = len(chardict.keys()) + 1
            batch.save_dictionary(chardict,charcount,'%s/dict.pkl' % save_path)
            # params
            params = init_params_c2w2s(n_chars=n_char)
        
        labeldict, labelcount = batch.build_label_dictionary(yt)
        batch.save_dictionary(labeldict, labelcount, '%s/label_dict.pkl' % save_path)

        n_classes = min(len(labeldict.keys()) + 1, MAX_CLASSES)

        # classification params
        params['W_cl'] = theano.shared(np.random.normal(loc=0., scale=SCALE, size=(SDIM,n_classes)).astype('float32'), name='W_cl')
        params['b_cl'] = theano.shared(np.zeros((n_classes)).astype('float32'), name='b_cl')

    else:
        print("Loading model params...")
        params = load_params_shared('%s/model.npz' % save_path)

        print("Loading dictionaries...")
        with open('%s/dict.pkl' % save_path, 'rb') as f:
            chardict = pkl.load(f)
        with open('%s/label_dict.pkl' % save_path, 'rb') as f:
            labeldict = pkl.load(f)
        n_char = len(chardict.keys()) + 1
        n_classes = min(len(labeldict.keys()) + 1, MAX_CLASSES)

    # iterators
    train_iter = batch.BatchTweets(Xt, yt, labeldict, batch_size=N_BATCH, max_classes=MAX_CLASSES)
    val_iter = batch.BatchTweets(Xv, yv, labeldict, batch_size=N_BATCH, max_classes=MAX_CLASSES)

    print("Building network...")

    # Tweet variables
    tweet = T.itensor4()
    targets = T.ivector()

    # masks
    t_mask = T.ftensor3()

    # network for prediction
    predictions, net, c2w, w2s = classify(tweet, t_mask, params, n_classes, n_char)

    # batch loss
    loss = lasagne.objectives.categorical_crossentropy(predictions, targets)
    cost = T.mean(loss) + REGULARIZATION*lasagne.regularization.regularize_network_params(net, lasagne.regularization.l2) + REGULARIZATION*lasagne.regularization.regularize_network_params(c2w, lasagne.regularization.l2)
    cost_only = T.mean(loss)
    reg_only = REGULARIZATION*lasagne.regularization.regularize_network_params(net, lasagne.regularization.l2) + REGULARIZATION*lasagne.regularization.regularize_network_params(c2w, lasagne.regularization.l2)

    # params and updates
    print("Computing updates...")
    lr = LEARNING_RATE
    mu = MOMENTUM
    updates = lasagne.updates.nesterov_momentum(cost, lasagne.layers.get_all_params(net), lr, momentum=mu)

    # Theano function
    print("Compiling theano functions...")
    inps = [tweet,t_mask,targets]
    #pred = theano.function([tweet,t_mask],predictions)
    #l = theano.function(inps,loss)
    cost_val = theano.function(inps,[cost_only,lasagne.layers.get_output(w2s)])
    train = theano.function(inps,cost,updates=updates)
    reg_val = theano.function([],reg_only)

    # Training
    print("Training...")
    uidx = 0
    start = time.time()
    try:
	for epoch in range(num_epochs):
	    n_samples = 0
            train_cost = 0.
	    print("Epoch {}".format(epoch))

            # learning schedule
            if epoch > 0 and epoch % 5 == 0:
                print("Updating Schedule...")
                lr = max(1e-5,lr/2)
                mu = mu - 0.05
                updates = lasagne.updates.nesterov_momentum(cost, lasagne.layers.get_all_params(net), lr, momentum=mu)
                train = theano.function(inps,cost,updates=updates)

            ud_start = time.time()
	    for xr,y in train_iter:
		n_samples +=len(xr)
		uidx += 1
                if DEBUG and uidx > 3:
                    sys.exit()

                if DEBUG:
                    print("Tweets = {}".format(xr[:5],y[:5]))
		x, x_m = batch.prepare_data_c2w2s(xr, chardict, n_chars=n_char)

		if x==None:
		    print("Minibatch with zero samples under maxlength.")
		    uidx -= 1
		    continue

                if DEBUG:
                    print("Params before update...")
                    print_params(params)
                    display_actv(x,x_m,y,tweet,t_mask,targets,net,'before')
                    cb, embb = cost_val(x,x_m,y)

		curr_cost = train(x,x_m,y)

                if DEBUG:
                    print("Params after update...")
                    print_params(params)
                    display_actv(x,x_m,y,tweet,t_mask,targets,net,'after')
                    ca, emba = cost_val(x,x_m,y)
                    print("Embeddings before = {}".format(embb[:5]))
                    print("Embeddings after = {}".format(emba[:5]))
                    print("Cost before update = {} \nCost after update = {}".format(cb, ca))

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
            validation_cost = 0.
            n_val_samples = 0
	    for xr,y in val_iter:
		n_val_samples +=len(xr)

		x, x_m = batch.prepare_data_c2w2s(xr, chardict, n_chars=n_char)

		if x==None:
                    print("Validation: Minibatch with zero samples under maxlength.")
		    continue

                vc, _ = cost_val(x,x_m,y)
                validation_cost += vc*len(xr)

            regularization_cost = reg_val()
	    print("Epoch {} Training Cost {} Validation Cost {} Regularization Cost {}".format(epoch, train_cost/n_samples, validation_cost/n_val_samples, regularization_cost))
	    print("Seen {} samples.".format(n_samples))

            for kk,vv in params.iteritems():
                print("Param {} Epoch {} Max {} Min {}".format(kk, epoch, np.max(vv.get_value()), np.min(vv.get_value())))

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
