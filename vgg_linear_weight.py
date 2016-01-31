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
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import scale
from scipy.spatial.distance import cosine
from sklearn.grid_search import GridSearchCV
#ROI name
roiName = ['other', 'V1', 'V2', 'V3', 'V3A', 'V3B', 'V4', 'LOC'];
# set parameters
targetROI = 6
layername = sys.argv[1]
print('training on input layer'+layername)
respPath = 'vgg_places_response/'
# load data
print('Loading fMRI Responses...')
f = h5py.File('../EstimatedResponses.mat', 'r')
data=f['/dataTrnS1'][()]
validData = f['/dataValS1'][()]
ROI = f['/roiS1'][()]
V1idx = np.nonzero(ROI==targetROI)[1] # pick V1
#V1idx = np.nonzero(np.logical_or(ROI==1, ROI==2))[1]
V1Resp = data[:,V1idx]
V1ValResp = validData[:,V1idx]
f.close()
# take off NaN pixels
tt = np.sum(np.isnan(V1Resp), axis = 0)
goodVoxel = np.nonzero(tt == 0)[0]
print('has '+str(goodVoxel.shape[0])+' voxels')
V1Resp = V1Resp[:, goodVoxel]
V1ValResp = V1ValResp[:, goodVoxel]

# load activation maps
print('Loading VGG representations...')
t = time.time()
for segidx in np.arange(16):
	datatype = 'Trn'
	if segidx == 0:
		fname = '../'+respPath+'seg'+str(segidx)+'_'+layername+'Val.h5'
		datatype = 'Val'
		with h5py.File(fname, 'r') as f:
			valfeat = f['feat'][()]
	else :
		fname = '../'+respPath+'seg'+str(segidx)+'_'+layername+'Trn.h5'
		with h5py.File(fname, 'r') as f:
			if 'trnfeat' in locals():
				trnfeat = np.append(trnfeat, f['feat'][()], axis = 0)
			else:
				trnfeat = f['feat'][()]
trnfeat = np.reshape(trnfeat, (trnfeat.shape[0], -1))
valfeat = np.reshape(valfeat, (valfeat.shape[0], -1))
print('load vgg time is '+str(time.time() - t))

trnX = trnfeat
trnY = V1Resp
valX = valfeat
valY = V1ValResp

print('train X has shape: ', trnX.shape)
print('train Y has shape: ', trnY.shape)
print('valid X has shape: ', valX.shape)
print('valid Y has shape: ', valY.shape)
# start training
t = time.time()
print('start training from vgg response to fMRI...')
nb_classes = V1ValResp.shape[1]
alpha = np.logspace(0, 9, num=10)
acc = np.zeros((len(alpha), nb_classes))
trn_acc = np.zeros((len(alpha), nb_classes))

preN = '../vgg_places_results_weight/'+roiName[targetROI]+layername
postN = ''
#np.save(preN+'seq'+postN, randSeq)
f_trn_acc = np.memmap(preN+'trn'+postN, dtype=trn_acc.dtype,shape=(nb_classes), mode='w+')
f_acc = np.memmap(preN+'acc'+postN, dtype=acc.dtype,shape=(nb_classes), mode='w+')
bestPred = np.memmap(preN+'bestpred'+postN, dtype='float32',shape=valY.shape, mode='w+')
bestTrain = np.memmap(preN+'besttrain'+postN, dtype='float32',shape=trnY.shape, mode='w+')
bestCoef = np.memmap(preN+'bestmodel'+postN, dtype='float32', shape=(trnX.shape[1], nb_classes), mode='w+')
bestOffset = np.memmap(preN+'bestcoef'+postN, dtype='float32', shape=(nb_classes), mode='w+')
t = time.time()
#def modelEstimate(j, f_trn_acc, f_acc, bestPred, bestTrain):
def modelEstimate(j):
	parameters = {'alpha':alpha}
	ridge = Ridge()
	model = GridSearchCV(ridge, parameters, cv=2)
	model.fit(trnX, trnY[:,j])
	y_pred = model.predict(valX)
	y_trn = model.predict(trnX)
	valscore = np.corrcoef(valY[:,j], y_pred)[0,1]
	trnscore = np.corrcoef(trnY[:,j], y_trn)[0,1]
	f_trn_acc[j] = trnscore
	f_acc[j] = valscore
	bestPred[:,j] = y_pred
	bestTrain[:,j] = y_trn
	qq = model.best_estimator_
	bestCoef[:,j] = qq.coef_
	bestOffset[j] = qq.intercept_
	#newpred = np.dot(valX, qq.coef_) + qq.intercept_
	print('feature '+str(j)+' has valscore '+"{0:.3f}".format(valscore))
#print('segment time is '+str(time.time()-t))
#for j in xrange(100):
#	for i in xrange(len(alpha)):
#Parallel(n_jobs = 20)(delayed(modelEstimate)(j, f_trn_acc, f_acc, bestPred, bestTrain) for j in range(nb_classes))
Parallel(n_jobs = 20)(delayed(modelEstimate)(j) for j in range(nb_classes))
print('segment time is '+str(time.time()-t))
#print(f_trn_acc[0,:10])
#print(f_acc[0,:10])
#np.save('trn_acc', trn_acc)
#np.save('acc', acc)
