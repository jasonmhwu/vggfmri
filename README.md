# this is the documentation of all codes
# MHWu, 10/01/2015

multiGaborRegress.py -> ridge regression on the multiscale Gabor activation map
plotAcc.py -> reads the learned regression result, mostly for debugging
plotAcc_norand.py -> reads the learned rf result, and calculates acc & corr
plotAcc_norand_featImp.py -> reads the learned rf result, and calculates acc & corr, also plot receptive field
multiGaborConv.py -> do convolution on images, multiscale Gabor, also contains deconv results
multiGaborDeconv.py -> do deconvolution to see results

Gabor2Response.py -> use Gabor or more as input feature to learn fmri response
Gabor2Response_rf.py -> use Gabor as input feature to learn fmri response by random forest regressor
GaborNonlinearTest.py -> use Gabor and its product to learn fmri
Label2Response_rf.py -> use label 1toN vector to learn fmri response
Label2Response_xfold.py -> use label 1toN vector to learn fmri response, with cross-validation
plotLabelResult.py -> check the distributions


======== MODEL ============
*.multiGabor -> (1750, 120), onlyV1
*.multiGaborV1V2 -> (1750, 120), V1V2
*.multi_1500_rand -> (1500, 250), V1V2

======= LABELDATA ========
in labeldata folder


