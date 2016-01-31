# check if forward label make senses

from __future__ import absolute_import
from __future__ import print_function
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import scipy.io, scipy.misc, h5py
import matplotlib.pyplot as plt
import time
import sys
import pdb
#sys.path.insert(0, '/home/mhwu/Documents/MATLAB/toolbox/liblinear-2.0/python')
#from liblinearutil import *
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import scale
from scipy.spatial.distance import cosine

trainNet = 'places'
layername = 'prob' 
caffe_root = '/home/aracity/caffe/'
responsePath = 'vgg_places_response/'
# load activation maps
print('Loading VGG representations...')
t = time.time()
for segidx in np.arange(16):
	datatype = 'Trn'
	if segidx == 0:
		fname = '../'+responsePath+'seg'+str(segidx)+'_'+layername+'Val.h5'
		datatype = 'Val'
		with h5py.File(fname, 'r') as f:
			valfeat = f['feat'][()]
	else :
		fname = '../'+responsePath+'/seg'+str(segidx)+'_'+layername+'Trn.h5'
		with h5py.File(fname, 'r') as f:
			if 'trnfeat' in locals():
				trnfeat = np.append(trnfeat, f['feat'][()], axis = 0)
			else:
				trnfeat = f['feat'][()]
trnfeat = np.reshape(trnfeat, (trnfeat.shape[0], -1))
valfeat = np.reshape(valfeat, (valfeat.shape[0], -1))
print(trnfeat.shape)
print(valfeat.shape)
print('load vgg time is '+str(time.time() - t))

# print label

if trainNet == 'imageNet':
	net_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
if trainNet == 'places':
	net_labels_filename = caffe_root + 'models/Places205-VGG/category.txt'

labels = np.loadtxt(net_labels_filename, str, delimiter='\t')

# sort top k predictions from softmax output
for i in range(1750):
	top_k = trnfeat[i].flatten().argsort()[-1:-6:-1]
	print('{0:20}  {1:.2f}  {2:20}  {3:.2f}'.format(labels[top_k[0]], trnfeat[i, top_k[0]], labels[top_k[1]], trnfeat[i, top_k[1]]))
for i in range(120):
	top_k = valfeat[i].flatten().argsort()[-1:-6:-1]
	print('{0:20}  {1:.2f}  {2:20}  {3:.2f}'.format(labels[top_k[0]], valfeat[i, top_k[0]], labels[top_k[1]], valfeat[i, top_k[1]]))
