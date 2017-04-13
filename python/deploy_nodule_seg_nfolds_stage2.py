import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import utils
#import cv2
#import random
#import time
import os
import matplotlib.pylab as plt
#import scipy as sp

import h5py    
import models
import hashlib
from glob import glob
#from keras.preprocessing.image import ImageDataGenerator
#from image import ImageDataGenerator
#%% data path

# luna data
path2luna="/media/mra/win71/data/misc/kaggle/datascience2017/LUNA2016/"
path2luna_nodes="/media/mra/My Passport/Kaggle/datascience2017/hpz440/LUNA2016/hdf5/nodes/"

# dsb data
root_data='/media/mra/win71/data/misc/kaggle/datascience2017/data/'
#path2dsb=root_data+'dsb.hdf5'
#path2dsbtest=root_data+'dsbtest.hdf5'
path2dsbtest=root_data+'stage2.hdf5'

# stage1 labels
path2csv=root_data+'stage1_labels.csv'


path2luna="/media/mra/My Passport/Kaggle/datascience2017/hpz440/LUNA2016/hdf5/subsets/"

subset_list=glob(path2luna+'subset*.hdf5')
subset_list.sort()
print 'total subsets: %s' %len(subset_list)


# logs
path2logs='./output/logs/'

#%%


# original size
H,W=512,512


# batch size
bs=8


# trained data dimesnsion
h,w=256,256

# input channel size
c_in=7

# seed point
seed = 2016
seed = np.random.randint(seed)
  

# number of outputs
nb_output=1

# log
now = datetime.datetime.now()
info='log_deploy_nfolds'
suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
# Direct the output to a log file and to screen
loggerFileName = os.path.join(path2logs,  suffix + '.txt')
utils.initialize_logger(loggerFileName)
#%%


# load train data
def load_luna_data(subsets):
    X=[]
    Y=[]
    for ss in subsets:
        print ss
        ff=h5py.File(ss,'r')
        for k in ff.keys():
            print k
            XY=ff[k]
            
            # find non-zeros masks/nodules, for every nodule three masks exist
            nz_Y=np.where(np.sum(XY[1],axis=(1,2))>0)[0]
            # pick the center masks
            nz_Y=nz_Y[1::3]
            #print nz_Y
            step=int(c_in/2)
            X0=[]
            Y0=[]
            for z in nz_Y:
                #print XY.shape[1]
                if z+step+1<XY.shape[1]:
                    XY1=XY[:,z-step:z+step+1]
                else:
                    XY1=XY[:,z-2*step:]
                X0.append(XY1[0])
                Y0.append(XY1[1][step])
            X0=np.stack(X0)    
            Y0=np.stack(Y0)
            Y0=Y0[:,np.newaxis,:]
            #print X0.shape
            #print Y0.shape
            X.append(X0)
            Y.append(Y0)
            
        ff.close()    
    X=np.vstack(X)        
    Y=np.vstack(Y).astype('uint8')        
    return X,Y


#%%
print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30)

########## load DSB data
df_train = pd.read_csv(path2csv)
print('Number of training patients: {}'.format(len(df_train)))
print('Cancer rate: {:.4}%'.format(df_train.cancer.mean()*100))
df_train.head()

# extract non cancers and non-cancers
non_cancer=df_train[df_train.cancer==0].id
cancer=df_train[df_train.cancer==1].id
print 'total non cancer:%s, total cancer:%s' %(len(non_cancer),len(cancer))

#%%

# fold number
foldnms=[5,6,7,8,9]
    
for foldnm in foldnms:
    print 'fold ', foldnm
    # experiment
    experiment='fold'+str(foldnm)+'_luna_seg'+'_hw_'+str(h)+'by'+str(w)+'_cin'+str(c_in)
    print ('experiment:', experiment)
    
    # weights
    weightfolder='./output/weights/'+experiment
    if  not os.path.exists(weightfolder):
        print ('weights folder does not exist!')
        raise IOError
    else:
        path2segweights=weightfolder+"/weights.hdf5"
        ff_w=h5py.File(path2segweights,'r')
        w0=ff_w[ff_w.keys()[0]]
        print 'cnn 1 shape:', (w0[w0.keys()[0]].shape)
        nb_filters=w0[w0.keys()[0]].shape[0]
        ff_w.close()

    # nodule segmentation network
    # training params
    params_seg={
        'h': h,
        'w': w,
        'c_in': c_in,           
        'weights_path': None,        
        'learning_rate': 3e-5,
        'optimizer': 'Adam',
        'loss': 'dice',
        'nbepoch': 1000,
        'nb_output': 1,
        'nb_filters': nb_filters,
        'max_patience': 30    
        }

    # model: nodule segmentation
    seg_model=models.seg_model(params_seg)
    seg_model.summary()

    # path to weights
    path2segweights=weightfolder+"/weights.hdf5"
    if  os.path.exists(path2segweights):
        seg_model.load_weights(path2segweights)
        print 'weights loaded!'
    else:
        raise IOError('weights not found!')
    
    weights_checksum=hashlib.md5(open(path2segweights, 'rb').read()).hexdigest()
    print 'checksum:', weights_checksum
    with open(weightfolder+'/checksum.csv', 'w+') as f:
        f.write(weights_checksum + '\n')    
    
    
    # verify network
    # pre-processing 
    param_prep={
        'h': h,
        'w': w,
        'crop'    : None,
        'norm_type' : 'minmax_bound',
        'output' : 'mask',
    }


    ################################### get test data
    ss_test=subset_list[foldnm]
    print 'test:', ss_test
    
    # path to nfold train and test data
    path2luna_train_test=path2luna+'fold'+str(foldnm)+'_train_test.hdf5'
    if os.path.exists(path2luna_train_test):
        ff_r=h5py.File(path2luna_train_test,'r')
        X_test=ff_r['X_test']
        Y_test=ff_r['Y_test']
        print 'hdf5 loaded '
    else:    
        # load images and masks    
        X_test,Y_test=load_luna_data([ss_test])   
        ff_w=h5py.File(path2luna_train_test,'w-')
        ff_w['X_test']=X_test
        ff_w['Y_test']=Y_test
        ff_w.close()
        print 'hdf5 saved!'
   
    #    path2luna_train_test=path2luna+'fold'+str(foldnm)+'_train_test.hdf5'
    #    if os.path.exists(path2luna_train_test):
    #        ff_r=h5py.File(path2luna_train_test,'r')
    #        X_test=ff_r['X_test']
    #        Y_test=ff_r['Y_test']
    #        print 'hdf5 loaded '
    #    else:
    #        raise IOError('train_test.hdf5 not found!')    

    # preprocess test data
    X,Y=utils.preprocess_XY(X_test,Y_test,param_prep)
    utils.array_stats(X)
    utils.array_stats(Y)
    print '-'*50

    # evaluate on test data for verification
    score_test=seg_model.evaluate(*utils.preprocess_XY(X_test,Y_test,param_prep),verbose=0)
    print ('score_test: %.2f' %(score_test[1]))
    print '-'*50
    

    ############################# obtain nodules for dsb test stage 2
    path2dsbtest_output=path2luna_nodes+'/fold'+str(foldnm)+'_stage2_nodes.hdf5'
    
    if not os.path.exists(path2dsbtest_output):
        f3=h5py.File(path2dsbtest_output,'w-')
    
        # read dsb hdf5
        f2=h5py.File(path2dsbtest,'r')
    
        for k1,id in enumerate(f2.keys()):
            print k1,id
            
            # read data 
            grp=f2[id]
            X1=grp['X']
            
            # read label    
            #y1=f2[id].attrs['cancer']
            
            # collect as N*c_in*H*W
            X2=[]
            for k in range(0,X1.shape[0]-c_in,3):
                X2.append(X1[k:k+c_in])
            X2=np.stack(X2)            
            utils.array_stats(X2)
        
            # preprocess        
            X2=utils.preprocess(X2,param_prep)
                 
            # obtain nodules as float32 
            Y=seg_model.predict(X2)
        
            # sum over masks
            sumYp=np.sum(Y>0.5,axis=(1,2,3))
            
            # number of non_zeros
            numofnzY=np.count_nonzero(sumYp)    
            
            # non-zero indices
            nzY_inds=np.nonzero(sumYp)    
            
            # sort by area
            sumYp_sort=np.argsort(-sumYp)
    
            print 'number of non-zero masks: %s' %numofnzY        
            #print 'non-zero indices: %s' %nzY_inds        
            #print 'top areas: %s' %sumYp_sort[:len(nzY_inds[0])]        
            
            # store nodules
            grp=f3.create_group(id)
            grp['Y']=Y # mask
            grp['cnz']=numofnzY # number of non-zero masks
            grp['nzYi']=nzY_inds # non zero indices
            grp['sYi']=sumYp_sort # sorted indices by area
            
        f2.close()
        f3.close()
    else:
        print 'dsb test nodules exist!'
    
#%%

