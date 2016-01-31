import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
import os
import scipy.io, scipy.misc, h5py, time, pdb
from os import listdir
from os.path import isfile, join
get_ipython().magic(u'pylab')
caffe_root = '/home/aracity/caffe/'
sys.path.insert(0, caffe_root + 'python')

# set parameters
writeResponse = 1
loadNet = 'placesVGG' # or 'origVGG'
imgLoadMethod = 'jpg' # or use 'mat'
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Switch Caffe between CPU and GPU mode, load the net in the test phase for inference, and configure input preprocessing.
# caffe.set_mode_cpu()
caffe.set_device(0)
caffe.set_mode_gpu()
if loadNet == 'origVGG':
	net = caffe.Net(caffe_root + '/models/vgg/VGG_ILSVRC_16_layers_deploy.prototxt',
                caffe_root + '/models/vgg/VGG_ILSVRC_16_layers.caffemodel',
                caffe.TEST)

if loadNet == 'placesVGG':
	net = caffe.Net(caffe_root + '/models/Places205-VGG/deploy_10.prototxt',
                caffe_root + '/models/Places205-VGG/snapshot_iter_765280.caffemodel',
                caffe.TEST)

# load images from jpg files
if imgLoadMethod == 'jpg':
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))
	if loadNet == 'origVGG':
		transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
	else:
		transformer.set_mean('data', np.array([105.487823486, 113.741088867, 116.060394287])) # mean pixel
		
	transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
	transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

	validFigPath = '../validFigFullRes/'
	trainFigPath = '../trainFigFullRes/'
	# valid
	print('forwarding validation set...')
	validfiles = [f for f in listdir(validFigPath) if isfile(join(validFigPath, f))]
	validfiles.sort()
	for fname in validfiles:
		if 'valData' in locals():
			valData = np.append(valData, transformer.preprocess('data', caffe.io.load_image(validFigPath+fname))[None,:], axis=0)
		else:
			valData = transformer.preprocess('data', caffe.io.load_image(validFigPath+fname))[None,:]
	net.blobs['data'].reshape(valData.shape[0],3,224,224)
	net.blobs['data'].data[...] = valData
	print(valData.shape)
	out = net.forward()
	for name, tmp in net.blobs.items():
		ftname = '../vgg_places_response/seg0_'+name+'Val.h5'
		if writeResponse:
			with h5py.File(ftname, 'w') as f:
				f.create_dataset("feat", data = net.blobs[name].data)
	# train
'''	
	print('forwarding training set...')
	trainfiles = [f for f in listdir(trainFigPath) if isfile(join(trainFigPath, f))]
	trainfiles.sort()
	for fname in trainfiles:
		if 'trnData' in locals():
			trnData = np.append(trnData, transformer.preprocess('data', caffe.io.load_image(trainFigPath+fname))[None,:], axis=0)
		else:
			trnData = transformer.preprocess('data', caffe.io.load_image(trainFigPath+fname))[None,:]
	SEGMENT = 120
	for segCnt in np.arange(trnData.shape[0]/SEGMENT+1):
		print('processing seg '+str(segCnt))
		trnSeg = trnData[SEGMENT*segCnt:SEGMENT*(segCnt+1)]
		net.blobs['data'].reshape(trnSeg.shape[0],3,224,224)
		net.blobs['data'].data[...] = trnSeg
		out = net.forward()
		for name, tmp in net.blobs.items():
			ftname = '../vgg_places_response/seg'+str(segCnt+1)+'_'+name+'Trn.h5'
			if writeResponse:
				with h5py.File(ftname, 'w') as f:
					f.create_dataset("feat", data = net.blobs[name].data)
'''
# set net to batch size of 50, load image
if imgLoadMethod == 'mat':
	IMGSIZE=224
	for fidx in np.arange(15, 16):
		print('processing mat '+str(fidx))
		datatype = 'Trn'
		if fidx ==0:
			fname = '../Stimuli_Val_FullRes.mat'
			datatype='Val'
		else:
			fname = '../Stimuli_Trn_FullRes_'+'{0:02}'.format(fidx)+'.mat'
		with h5py.File(fname, 'r') as f:
			data = f['/stim'+datatype][()].transpose(2,1,0)
		# resize to IMG_SIZE and scale to 1
		data_resized = np.zeros((data.shape[0], 3, IMGSIZE, IMGSIZE))
		for imgidx in np.arange(data.shape[0]):
			for coloridx in np.arange(3):
				data_resized[imgidx, coloridx] = scipy.misc.imresize((data[imgidx]+0.55)*255, (IMGSIZE, IMGSIZE)) # scale: 0~255

		net.blobs['data'].reshape(data_resized.shape[0],3,IMGSIZE, IMGSIZE)
		net.blobs['data'].data[...] = data_resized
		out = net.forward()

		for name, tmp in net.blobs.items():
			ftname = '../vgg_response/seg'+str(fidx)+'_'+name+datatype+'.h5'
			if writeResponse:
				with h5py.File(ftname, 'w') as f:
					f.create_dataset("feat", data = net.blobs[name].data)
	# get responses
#	feat = net.blobs['conv1_1'].data

# plot original image
# plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))

# show architecture
[(k, v.data.shape) for k, v in net.blobs.items()]

# The parameters and their shapes. The parameters are `net.params['name'][0]` while biases are `net.params['name'][1]`.
[(k, v[0].data.shape) for k, v in net.params.items()]


# Helper functions for visualization
# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)

def plotOrig(idx):
	plt.imshow(transformer.deprocess('data', net.blobs['data'].data[idx]))

feat = net.blobs['conv1_2'].data[1,:36]
vis_square(feat, padval=1)
