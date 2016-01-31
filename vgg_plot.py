# some plotting helper functions for vgg
import numpy as np
import matplotlib.pyplot as plt
import pdb

# dictionaries
NUMVOX = {'V4':1535, 'LOC':928}
NUMFEAT= {'pool4':100352, 'pool5':25088}

def readResult(preN, roi, layer):
	postN = ''
	numVox = NUMVOX[roi]
	featureNum = NUMFEAT[layer]
	numAlpha = 10
	f_trn_acc = np.memmap(preN+'trn'+postN,dtype = 'float64', shape=(numVox), mode='r')
	f_acc = np.memmap(preN+'acc'+postN, dtype = 'float64', shape=(numVox), mode='r')
	bestPred = np.memmap(preN+'bestpred'+postN, dtype='float32',shape=(120, numVox), mode='r')
	bestTrain = np.memmap(preN+'besttrain'+postN, dtype='float32',shape=(1750, numVox), mode='r')
	bestModel = np.memmap(preN+'bestmodel'+postN, dtype='float32',shape=(featureNum, numVox), mode='r')
	bestCoef = np.memmap(preN+'bestcoef'+postN, dtype='float32',shape=(numVox), mode='r')
	return f_trn_acc, f_acc, bestPred, bestTrain

def plotMaxAcc(trn1, val1, trn2, val2):
#	pdb.set_trace()
	plt.figure(1)
	plt.plot(val1)
	plt.figure(2)
	plt.plot(val2)
	plt.show()
def plotHist(arr):
	plt.hist(arr, bins = 20)
	plt.show()
def calcCorr(arr1, arr2):
	corr = np.corrcoef(arr1, arr2)
	print('the corr of 2 layers is '+str(corr[0,1]))

def plotScatter(arr1, arr2):
	plt.scatter(arr1, arr2)
	plt.show()
def main():
#	plotMaxAcc()
#	calcCorr()
	trn1, val1, tmp1, tmp2 = readResult('../vgg_places_results_weight/V4pool5', 'V4', 'pool5')
	trn2, val2, tmp3, tmp4 = readResult('../vgg_places_results_weight/V4pool4', 'V4', 'pool4')
	plotHist(val1)
	pdb.set_trace()
if __name__ == '__main__':
	main()
