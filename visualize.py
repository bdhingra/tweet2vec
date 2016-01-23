import cPickle as pkl
import numpy as np
import data_visualization
import sys
import random
import string
import re

pattern = re.compile('^[\w ]+$')

dpath = sys.argv[1]
vpath = sys.argv[2]

d = pkl.load(open(dpath,'r'))
v = pkl.load(open(vpath,'r'))
index = random.sample(range(len(d)), 1000)
d = [d[ii] for ii in index]
v = [v[ii] for ii in index]

vnp = np.zeros((len(v),len(v[0])))
for i in range(len(v)):
    vnp[i,:] = v[i]
vectors = np.asfarray( vnp, dtype='float' )

a = []
for item in d:
    a.append(pattern.sub('',item))

vis = data_visualization.Visualize(vectors, annotations=a)
vis.princomp(50)
vis.reset()
vis.tsne(2)
p = vis.scatter_plot()
p.show()
