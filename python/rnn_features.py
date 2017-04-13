#%% deploy classify nodes on DSB
import numpy as np
import cv2
import time
import os
import matplotlib.pylab as plt
import models
#import utils
#from keras import backend as K
#from keras.utils import np_utils
import h5py
#from glob import glob
#from sklearn import cross_validation
#import xgboost as xgb
#from mpl_toolkits.mplot3d import Axes3D
#from sklearn.decomposition import PCA
#from sklearn.cluster import KMeans
import pandas as pd    
from sklearn.model_selection import train_test_split
#from keras.utils import np_utils
#from skimage import measure
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
#from keras.models import load_model
#%%
# settings

# path to dataset
#path2luna_external="/media/mra/My Passport/Kaggle/datascience2017/hpz440/LUNA2016/hdf5/"
#path2subsets=path2luna_external+"subsets/"
#path2chunks=path2luna_external+"chunks/"
#path2luna_internal="/media/mra/win71/data/misc/kaggle/datascience2017/LUNA2016/"
#path2chunks=path2luna_internal+"chunks/"

#path2dsb_external='/media/mra/My Passport/Kaggle/datascience2017/hpz440/data/hdf5/'
path2dsb_internal="/media/mra/win71/data/misc/kaggle/datascience2017/data/"

#subset_list=glob(path2chunks+'subset*.hdf5')
#subset_list.sort()
#print 'total subsets: %s' %len(subset_list)


pre_train=False
#%%

    
# conver chunk number to region
def chunknb2zyx(chunk_nb,(N,H,W,step)):
    n=np.ceil(N*1./step)
    h=np.ceil(H*1./step)
    w=np.ceil(W*1./step)
    nb_chunks=n*h*w
    if chunk_nb>nb_chunks:
        raise IOError
    z,r=divmod(chunk_nb,(h*w))
    y,x=divmod(r,(h))
        
    #    if (x<=w/2) & (y<=w/2):
    #        loc=0 #'top left'
    #    elif (x>w/2) & (y<=w/2):
    #        loc=1 #'top right'
    #    elif (x<=w/2) & (y>w/2):
    #        loc=2 #'bottom left'        
    #    elif (x>w/2) & (y>w/2):
    #        loc=3 #'bottom right'        
    #print z,y,x    
    return z,y,x
    #return (z*step,(y+1)*step,(x+1)*step)

#%%

# loading features from dsb data set
path2dsb_features=path2dsb_internal+'dsb_features.hdf5'
if os.path.exists(path2dsb_features):
    ff_r=h5py.File(path2dsb_features,'r')
    print len(ff_r)
else:
    print 'does not exist'

# read dsb resample
#path2dsb_resample=path2dsb_internal+'dsb_resampleXY.hdf5'
#ff_dsb=h5py.File(path2dsb_resample,'r')
#step=24


#%%    

# loading stage 1 labels
df_train = pd.read_csv('../stage1_labels.csv')
print('Number of training patients: {}'.format(len(df_train)))
print('Cancer rate: {:.4}%'.format(df_train.cancer.mean()*100))
df_train.head()
y_cancer = df_train['cancer'].as_matrix()

#%%

# define model
params_train={
        'timesteps': 130,
        'input_length': 512,
        'nb_gru': 50,
        'nb_dense': 128,
        'loss': 'binary_crossentropy',
        'nbepoch': 200,
        'lr': 1e-4,
        'max_patience': 50,
        }

# model        
model=models.model_rnn(params_train)
model.summary()

# exeriment name to record weights and scores
experiment='RNN_features'
print ('experiment:', experiment)

# checkpoint
weightfolder='./output/weights/'+experiment
if  not os.path.exists(weightfolder):
    os.makedirs(weightfolder)
    print ('weights folder created')

# path to weights
path2weights=weightfolder+"/weights.hdf5"

#%%

print 'wait to extract features ....'
path2dsb_nzfeatures=path2dsb_internal+'dsb_nzfeatures.hdf5'


if not os.path.exists(path2dsb_nzfeatures):
    nz_features=[]
    for k,id in enumerate(df_train.id):
        print k,id,y_cancer[k]
    
        # read group features and labels    
        grp=ff_r[id]
        feature=grp['features']#.value
        # max pool over 2*2*2
        feature=np.max(feature,axis=(2,3,4))
        
        # get output
        y_pred=np.argmax(grp['output'],axis=1)
        #sort_i=np.argsort(y_pred)        
        nz_i=np.nonzero(y_pred)
    
        # non-zero features
        nz_feature=feature[nz_i]   
        
        # zero padding
        nz_feature=np.append(nz_feature,np.zeros((130-nz_feature.shape[0],nz_feature.shape[1])),axis=0)
        
        #collect all non-zero features
        nz_features.append(nz_feature)

    # create hdf5 to store non zero features
    ff_dsbnz=h5py.File(path2dsb_nzfeatures,'w-')
    ff_dsbnz['nz_features']=nz_features
    ff_dsbnz.close()
else:
    # read non-zero features
    ff_dsbnz=h5py.File(path2dsb_nzfeatures,'r')
    nz_features=ff_dsbnz['nz_features']

# convert to numpy array
x=np.stack(nz_features)

x_train, x_test, y_train, y_test = train_test_split(x, y_cancer, random_state=42,stratify=y_cancer, 
                                                               test_size=0.20)
    
        
#%%

print ('training in progress ...')

# checkpoint settings
#checkpoint = ModelCheckpoint(path2weights, monitor='val_loss', verbose=0, save_best_only='True',mode='min')

# load last weights
if pre_train:
    if  os.path.exists(path2weights):
        model.load_weights(path2weights)
        print 'weights loaded!'
    else:
        raise IOError('weights not exist!')

# path to csv file to save scores
path2scorescsv = weightfolder+'/scores.csv'
first_row = 'train,test'
with open(path2scorescsv, 'w+') as f:
    f.write(first_row + '\n')
    
    
# Fit the model
start_time=time.time()
scores_test=[]
scores_train=[]
if params_train['loss']=='dice': 
    best_score = 0
    previous_score = 0
else:
    best_score = 1e6
    previous_score = 1e6
patience = 0


# convert class vectors to binary class matrices
#y_train = np_utils.to_categorical(trn_y, num_classes)
#y_test = np_utils.to_categorical(val_y, num_classes)


for epoch in range(params_train['nbepoch']):

    print ('epoch: %s,  Current Learning Rate: %.1e' %(epoch,model.optimizer.lr.get_value()))
    seed = np.random.randint(0, 999999)

    hist=model.fit(x_train, y_train, nb_epoch=1, batch_size=32,verbose=0,shuffle=True)
    
    # evaluate on test and train data
    score_test=model.evaluate(x_test,y_test,verbose=0,batch_size=32)
    score_train=hist.history['loss']
   
    print ('score_train: %s, score_test: %s' %(score_train,score_test))
    scores_test=np.append(scores_test,score_test)
    scores_train=np.append(scores_train,score_train)    

    # check if there is improvement
    if params_train['loss']=='dice': 
        if (score_test>=best_score):
            print ("!!!!!!!!!!!!!!!!!!!!!!!!!!! viva, improvement!!!!!!!!!!!!!!!!!!!!!!!!!!!") 
            best_score = score_test
            patience = 0
            model.save_weights(path2weights)       
            model.save(weightfolder+'/model.h5')
            
        # learning rate schedule
        if score_test<=previous_score:
            #print "Incrementing Patience."
            patience += 1
    else:
        if (score_test<=best_score):
            print ("!!!!!!!!!!!!!!!!!!!!!!!!!!! viva, improvement!!!!!!!!!!!!!!!!!!!!!!!!!!!") 
            best_score = score_test
            patience = 0
            model.save_weights(path2weights)  
            model.save(weightfolder+'/model.h5')
            
        # learning rate schedule
        if score_test>previous_score:
            #print "Incrementing Patience."
            patience += 1
    # save anyway    
    #model.save_weights(path2weights)      
            
    if patience == params_train['max_patience']:
        params_train['learning_rate'] = params_train['learning_rate']/2
        print ("Upating Current Learning Rate to: ", params_train['learning_rate'])
        model.optimizer.lr.set_value(params_train['learning_rate'])
        print ("Loading the best weights again. best_score: ",best_score)
        model.load_weights(path2weights)
        patience = 0
    
    # save current test score
    previous_score = score_test    
    
    # real time plot
    #plt.plot([e],[score_train],'b.')
    #plt.plot([e],[score_test],'b.')
    #display.clear_output(wait=True)
    #display.display(plt.gcf())
    #sys.stdout.flush()
    
    # store scores into csv file
    with open(path2scorescsv, 'a') as f:
        string = str([score_train,score_test])
        f.write(string + '\n')
       

print ('model was trained!')
elapsed_time=(time.time()-start_time)/60
print ('elapsed time: %d  mins' %elapsed_time)          

