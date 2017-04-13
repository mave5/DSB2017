#%% deploy classify nodes on DSB
import numpy as np
import cv2
import time
import os
import matplotlib.pylab as plt
import models
import utils
from keras import backend as K
from keras.utils import np_utils
import h5py
from glob import glob
from image import ImageDataGenerator
from keras.utils import np_utils
#from skimage import measure
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
from keras.models import load_model
#%%
# settings

# path to dataset
path2luna_external="/media/mra/My Passport/Kaggle/datascience2017/hpz440/LUNA2016/hdf5/"
path2subsets=path2luna_external+"subsets/"
path2chunks=path2luna_external+"chunks/"
#path2luna_internal="/media/mra/win71/data/misc/kaggle/datascience2017/LUNA2016/"
#path2chunks=path2luna_internal+"chunks/"

path2dsb_internal="/media/mra/win71/data/misc/kaggle/datascience2017/data/"

subset_list=glob(path2chunks+'subset*.hdf5')
subset_list.sort()
print 'total subsets: %s' %len(subset_list)

#%%

# pre-processed data dimesnsion
z,h,w=64,64,64

# batch size
bs=8

# number of classes
num_classes=11

# fold number
foldnm=2

# seed point
seed = 2017
seed = np.random.randint(seed)

# exeriment name to record weights and scores
#experiment='fold'+str(foldnm)+'_luna_classify_positive_negative'+'_hw_'+str(h)+'by'+str(w)+'_cin'+str(z)
experiment='fold'+str(foldnm)+'_luna_classify_positive_negative_bysize'+'_hw_'+str(h)+'by'+str(w)+'_cin'+str(z)
print ('experiment:', experiment)

# checkpoint
weightfolder='./output/weights/'+experiment
if  not os.path.exists(weightfolder):
    os.makedirs(weightfolder)
    print ('weights folder created')


# pre train
pre_train=True
#%%

########## log
import datetime
path2logs='./output/logs/'
now = datetime.datetime.now()
info='log_DeployClassifyPositiveNegative_'
suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
# Direct the output to a log file and to screen
loggerFileName = os.path.join(path2logs,  suffix + '.txt')
utils.initialize_logger(loggerFileName)


#%%

# load train data
def load_data(subsets):
    X=[]
    y=[]
    for ss in subsets:
        print ss
        ff=h5py.File(ss,'r')
        for k in ff.keys():
            print k
            X0=ff[k]['X'].value
            y0=ff[k]['y'].value
            X.append(X0)
            y.append(y0)
        ff.close()    
    X=np.vstack(X)    
    y=np.hstack(y)
    return X,y


def extract_chunks(X,w):
    # chunks
    N,H,W=X.shape    
    #step=w/2
    step=24
    
    X_chunks=[]
    nb_chunks=0
    for cz in range(0,N,step):
        for cy in range(0,H,step):
            for cx in range(0,W,step):
                nb_chunks+=1
                # extract over lapping chunks
                X_chunk=X[cz:cz+w,cy:cy+w,cx:cx+w]
                
                # check if it is w*w*w
                n1,h1,w1=X_chunk.shape    
                if w1<w:
                    n1,h1,w1=X_chunk.shape    
                    X_chunk=np.append(X_chunk,np.zeros((n1,h1,w-w1),'int16'),axis=2)                    
                if h1<w:
                    n1,h1,w1=X_chunk.shape    
                    X_chunk=np.append(X_chunk,np.zeros((n1,w-h1,w1),'int16'),axis=1)                    
                if n1<w:
                    n1,h1,w1=X_chunk.shape    
                    X_chunk=np.append(X_chunk,np.zeros((w-n1,h1,w1),'int16'),axis=0)                    
                if np.sum(X_chunk.shape)!= 3*w:
                    raise IOError('incompatible size!')
                # collect chunks    
                X_chunks.append(X_chunk)    
    X_chunks=np.stack(X_chunks)
    return X_chunks


#%%

print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30)

# path to nfold train and test data
path2luna_train_test=path2chunks+'fold'+str(foldnm)+'_train_test_chunks.hdf5'
if os.path.exists(path2luna_train_test):
    ff_r=h5py.File(path2luna_train_test,'r')
    #X_train=ff_r['X_train']
    #y_train=ff_r['y_train'].value
    X_test=ff_r['X_test']
    y_test=ff_r['y_test'].value
    print 'hdf5 loaded'


# convert class vectors to binary class matrices
y_test=np.round(y_test/4).astype('uint8')
y_test = np_utils.to_categorical(y_test, num_classes)
    
#%%
print('-'*30)
print('Creating and compiling model...')
print('-'*30)

# training params
params_train={
    'h': h,
    'w': w,
    'z': z,
    'c':1,           
    'learning_rate': 3e-5,
    'optimizer': 'Adam',
    #'loss': 'mean_squared_error',
    'loss': 'categorical_crossentropy',
    'nbepoch': 2000,
    'num_labels': num_classes,
    'nb_filters': 32,    
    'max_patience': 20    
        }

# path to weights
#model = models.model_3d_1(params_train)
#model = models.model_3d(params_train)

#path2weights=weightfolder+"/weights.hdf5"
#if  os.path.exists(path2weights):
    #model.load_weights(path2weights)
    #print 'weights loaded!'
#else:
    #raise IOError
#model.save(path2model)

path2model=weightfolder+"/model.hdf5"
if  os.path.exists(path2model):
    model=load_model(path2model)    
    print 'model loaded!'
else:
    raise IOError

model.summary()

score_test=model.evaluate(np.array(X_test)[:,np.newaxis],y_test,verbose=0,batch_size=8)
print ('score_test: %s' %(score_test))

#y_pred=model.predict(np.array(X_test)[:,np.newaxis],batch_size=8)

# feature extractor
#n=15
#get_nth_layer_output = K.function([model.layers[0].input],[model.layers[n].output])

#%%     
#c1=31
#n1=np.random.randint(X_test.shape[0])
#X1=X_test[n1,c1,:]
#plt.imshow(X1,cmap='gray')
#plt.title([y_test[n1],y_pred[n1]])
#plt.show()
#%%

# deploy on DSB data
path2dsb_features=path2dsb_internal+'dsb_features.hdf5'
if not os.path.exists(path2dsb_features):
    ff_dsb_features=h5py.File(path2dsb_features,'a')

    # path 2 dsb
    path2dsb_resample=path2dsb_internal+'dsb_resampleXY.hdf5'
    ff_dsb=h5py.File(path2dsb_resample,'r')
    
    # loop over dsb
    start_point=456
    for key in ff_dsb.keys()[start_point:]:
        print start_point,key
        start_point+=1
        start_time=time.time()
        
        # get image and lung mask
        X=np.array(ff_dsb[key]['X'],'float32')
        X=utils.normalize(X)
        Y=ff_dsb[key]['Y'] # lung mask
        
        # extract lung
        X=X*Y
        #utils.array_stats(X)
        
        # extract chunks
        X_chunks=extract_chunks(X,w)
        print X_chunks.shape
    
                   
        # feed to net
        #y_pred=model.predict(X_chunks[:,np.newaxis],batch_size=bs)   
        features=[]
        for k in range(0,X_chunks.shape[0],bs):
            #print k
            #tmp=get_nth_layer_output([X_chunks[k:k+bs][:,np.newaxis]])[0]
            tmp=model.predict(X_chunks[k:k+bs][:,np.newaxis])
            features.append(tmp)
        features=np.vstack(features)    
        
        # get average
        #features=np.mean(features,axis=0)        
        
        # write into file
        ff_dsb_features[key]=features
        print 'elapsed time: %.2f min' %((time.time()-start_time)/60.)
        
    ff_dsb_features.close()
else:
    'hdf5 esits!'

#%%

# deploy on DSB test data
path2dsbtest_features=path2dsb_internal+'dsbtest_features.hdf5'
ff_dsbtest_features=h5py.File(path2dsbtest_features,'a')

# path 2 dsb
path2dsbtest_resample=path2dsb_internal+'dsbtest_resampleXY.hdf5'
ff_dsbtest=h5py.File(path2dsbtest_resample,'r')

# loop over dsb
start_point=0
for key in ff_dsbtest.keys()[start_point:]:
    print start_point,key

    # get image and lung mask
    X=np.array(ff_dsbtest[key]['X'],'float32')
    X=utils.normalize(X)
    Y=ff_dsbtest[key]['Y'] # lung mask
    
    # extract lung
    X=X*Y
    #utils.array_stats(X)
    
    # extract chunks
    X_chunks=extract_chunks(X,w)
    print X_chunks.shape

               
    # feed to net
    #y_pred=model.predict(X_chunks[:,np.newaxis],batch_size=bs)   
    features=[]
    for k in range(0,X_chunks.shape[0],bs):
        #print k
        #tmp=get_nth_layer_output([X_chunks[k:k+bs][:,np.newaxis]])[0]
        tmp=model.predict(X_chunks[k:k+bs][:,np.newaxis])
        features.append(tmp)
    features=np.vstack(features)    
    
    # get average
    #features=np.mean(features,axis=0)        
    
    # write into file
    ff_dsbtest_features[key]=features
    
ff_dsbtest_features.close()
#%%

ff_dsbtest_features=h5py.File(path2dsbtest_features,'r')
print len(ff_dsbtest_features.keys())
for key in ff_dsbtest_features.keys():
    print key
    features=ff_dsbtest_features[key]
    print np.max(features)
    
Features=[]    
ff_dsb_features=h5py.File(path2dsb_features,'r')
print len(ff_dsbtest_features.keys())
for key in ff_dsb_features.keys():
    print key
    features=ff_dsb_features[key]
    print np.max(features)
    Features.append(np.max(features))

plt.hist(Features)