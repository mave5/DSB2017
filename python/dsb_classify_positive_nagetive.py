#%% classify positive nodes and negative nodes

import numpy as np
import cv2
import time
import os
import matplotlib.pylab as plt
#from skimage import measure
import models
import utils
from keras import backend as K
from keras.utils import np_utils
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
import h5py
from sklearn import cross_validation
import pandas as pd
from glob import glob
from image import ImageDataGenerator
from keras.utils import np_utils
#%%
# settings

# path to dataset
#path2luna_external="/media/mra/My Passport/Kaggle/datascience2017/hpz440/LUNA2016/hdf5/"
#path2subsets=path2luna_external+"subsets/"
#path2chunks=path2luna_external+"chunks/"
#path2luna_internal="/media/mra/win71/data/misc/kaggle/datascience2017/LUNA2016/"
#path2chunks=path2luna_internal+"chunks/"
path2dsb_internal="/media/mra/win71/data/misc/kaggle/datascience2017/data/"

#subset_list=glob(path2chunks+'subset*.hdf5')
#subset_list.sort()
#print 'total subsets: %s' %len(subset_list)

#%%

# pre-processed data dimesnsion
z,h,w=64,64,64

# batch size
bs=8

# number of classes
num_classes=2

# fold number
foldnm=1

# seed point
seed = 2017
seed = np.random.randint(seed)

# exeriment name to record weights and scores
experiment='fold'+str(foldnm)+'_dsb_classify_positive_negative_bysize'+'_hw_'+str(h)+'by'+str(w)+'_cin'+str(z)
print ('experiment:', experiment)

# checkpoint
weightfolder='./output/weights/'+experiment
if  not os.path.exists(weightfolder):
    os.makedirs(weightfolder)
    print ('weights folder created')

# data augmentation 
augmentation=True

# pre train
pre_train=True
#%%

########## log
import datetime
path2logs='./output/logs/'
now = datetime.datetime.now()
info='log_dsbClassifyPositiveNegative_'
suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
# Direct the output to a log file and to screen
loggerFileName = os.path.join(path2logs,  suffix + '.txt')
utils.initialize_logger(loggerFileName)


#%%

# random data generator
datagen = ImageDataGenerator(featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.01,
        zoom_range=0.01,
        channel_shift_range=0.0,
        fill_mode='constant',
        cval=0.0,
        horizontal_flip=True,
        vertical_flip=True,
        dim_ordering='th') 


def iterate_minibatches(inputs1 , targets,  batchsize, shuffle=True, augment=True):
    assert len(inputs1) == len(targets)
    if augment==True:
        if shuffle:
            indices = np.arange(len(inputs1))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs1) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            x = inputs1[excerpt]
            y = targets[excerpt] 
            for  xxt,yyt in datagen.flow(x, y , batch_size=x.shape[0]):
                x = xxt.astype(np.float32) 
                y = yyt 
                break
    else:
        x=inputs1
        y=targets

    #yield x, np.array(y, dtype=np.uint8)         
    return x, np.array(y, dtype=np.uint8)         

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


def extract_chunk(X,w,chunk_nb):
    # chunks
    N,H,W=X.shape    
    step=24
    
    cz,cy,cx=chunknb2zyx(chunk_nb,(N,H,W,step))
    #print cz,cy,cx
    
    # extract over lapping chunks
    cz=cz*step
    cx=cx*step
    cy=cy*step
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
    return X_chunk

def train_test_split(df_train):
    # extract non cancers
    non_cancer=df_train[df_train.cancer==0].id
    cancer=df_train[df_train.cancer==1].id
    print 'total non cancer:%s, total cancer:%s' %(len(non_cancer),len(cancer))
    
    y_nc=np.zeros(len(non_cancer),'uint8')
    y_c=np.ones(len(cancer),'uint8')

    # train val split
    path2trainvalindx=weightfolder+'/train_val_index.npz'
    if not os.path.exists(path2trainvalindx):
        trn_nc, val_nc, trn_ync, val_ync = cross_validation.train_test_split(non_cancer,y_nc, random_state=420, stratify=y_nc,test_size=0.1)                                                                   
        trn_c, val_c, trn_yc, val_yc = cross_validation.train_test_split(cancer,y_c, random_state=420, stratify=y_c,test_size=0.1) 
    
        # indices of train and validation
        trn_ids=np.concatenate((trn_nc,trn_c))
        trn_y=np.concatenate((trn_ync,trn_yc))
    
        val_ids=np.concatenate((val_nc,val_c))
        val_y=np.concatenate((val_ync,val_yc))
    
        # shuffle
        trn_ids,trn_y=utils.unison_shuffled_copies(trn_ids,trn_y)
        val_ids,val_y=utils.unison_shuffled_copies(val_ids,val_y)
        np.savez(path2trainvalindx,trn_ids=trn_ids,val_ids=val_ids,trn_y=trn_y,val_y=val_y)
        print 'train validation indices saved!'    
    else:
        f_trvl=np.load(path2trainvalindx)    
        trn_ids=f_trvl['trn_ids']
        trn_y=f_trvl['trn_y']    
        val_ids=f_trvl['val_ids']
        val_y=f_trvl['val_y']
        print 'train validation indices loaded!'

    return trn_ids,val_ids,trn_y,val_y

#%%

# read dsb resample
path2dsb_resample=path2dsb_internal+'dsb_resampleXY.hdf5'
ff_dsb=h5py.File(path2dsb_resample,'r')
step=24

path2dsb_features=path2dsb_internal+'dsb_features.hdf5'
if os.path.exists(path2dsb_features):
    ff_dsb_features=h5py.File(path2dsb_features,'r')
    print len(ff_dsb_features)
else:
    print 'does not exist'

# path to suspicious chunks
path2dsb_chunks=path2dsb_internal+'dsb_chunks.hdf5'

if not os.path.exists(path2dsb_chunks):

    ff_dsb_chunks=h5py.File(path2dsb_chunks,'w-')

    for k,key in enumerate(ff_dsb_features.keys()):
        print k,key
    
        # read chunk outputs
        out=ff_dsb_features[key]
        y_pred=np.argmax(out,axis=1)
        # sort output         
        sort_i=np.argsort(y_pred)        
        
        # get image and lung mask
        X=np.array(ff_dsb[key]['X'],'float32')
        X=utils.normalize(X)
        Y=ff_dsb[key]['Y'] # lung mask
        
        # extract lung
        X=X*Y
        #utils.array_stats(X)

        # get chunk location    
        N,H,W=X.shape
        step=24
        chunk_nb=sort_i[-1]
        chunknb2zyx(chunk_nb,(N,H,W,step))
        Xc=extract_chunk(X,w,chunk_nb)
        
        # extract chunks
        #X_chunks=extract_chunks(X,w)
        #print X_chunks.shape
        # bigest node
        #X0=X_chunks[sort_i[-1]]
        #print np.sum(Xc),np.sum(X0)
        
        # store chunk into hdf5
        ff_dsb_chunks[key]=Xc
    # close hdf5    
    ff_dsb_chunks.close()        
else:
    print 'loaded HDF5'
    ff_dsb_chunks=h5py.File(path2dsb_chunks,'r')
    print len(ff_dsb_chunks)
ddddd

df_train = pd.read_csv('../stage1_labels.csv')
print('Number of training patients: {}'.format(len(df_train)))
print('Cancer rate: {:.4}%'.format(df_train.cancer.mean()*100))
df_train.head()
  
# train validation split
trn_ids,val_ids,trn_y,val_y=train_test_split(df_train)

X_train=[]
for key in trn_ids:
    X_train.append(ff_dsb_chunks[key])
X_train=np.stack(X_train)

X_test=[]
for key in val_ids:
    X_test.append(ff_dsb_chunks[key])
X_test=np.stack(X_test)

utils.array_stats(X_train)    
utils.array_stats(X_test)    


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
    'nbepoch': 200,
    'num_labels': num_classes,
    'nb_filters': 16,    
    'max_patience': 20    
        }

model = models.model_3d(params_train)
model.summary()

# path to weights
path2weights=weightfolder+"/weights.hdf5"

#%%

print ('training in progress ...')

# checkpoint settings
#checkpoint = ModelCheckpoint(path2weights, monitor='val_loss', verbose=0, save_best_only='True',mode='min')

# load last weights
if pre_train:
    if  os.path.exists(path2weights) and pre_train:
        model.load_weights(path2weights)
        print 'weights loaded!'
    else:
        raise IOError

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
y_train = np_utils.to_categorical(trn_y, num_classes)
y_test = np_utils.to_categorical(val_y, num_classes)


for epoch in range(params_train['nbepoch']):

    print ('epoch: %s,  Current Learning Rate: %.1e' %(epoch,model.optimizer.lr.get_value()))
    seed = np.random.randint(0, 999999)

    # data augmentation
    bs2=16
    for k in range(0,len(trn_ids),bs2):
        #print k

        try:
            X_batch=X_train[k:k+bs2]#[:,np.newaxis]
            y_batch=y_train[k:k+bs2]
        except:
            print 'skept this batch!'
            continue
        
        # augmentation
        X_batch,_=iterate_minibatches(X_batch,X_batch,X_batch.shape[0],shuffle=False,augment=True)        
        hist=model.fit(X_batch[:,np.newaxis], y_batch, nb_epoch=1, batch_size=bs,verbose=0,shuffle=True)
        #print 'partial loss:', hist.history['loss']
    
    # evaluate on test and train data
    score_test=model.evaluate(X_test[:,np.newaxis],y_test,verbose=0,batch_size=bs)

    #if params_train['loss']=='dice': 
        #score_test=score_test[1]   
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
#%%

plt.figure(figsize=(15,10))
plt.plot(scores_test)
plt.plot(scores_train)
plt.title('train-validation progress',fontsize=20)
plt.legend(('test','train'), loc = 'lower right',fontsize=20)
plt.xlabel('epochs',fontsize=20)
plt.ylabel('loss',fontsize=20)
plt.grid(True)
plt.show()
plt.savefig(weightfolder+'/train_val_progress.png')

print ('best scores train: %.5f' %(np.max(scores_train)))
print ('best scores test: %.5f' %(np.max(scores_test)))          
          
#%%
# loading best weights from training session
print('-'*30)
print('Loading saved weights...')
print('-'*30)
# load best weights

if  os.path.exists(path2weights):
    model.load_weights(path2weights)
    print 'weights loaded!'
else:
    raise IOError


score_test=model.evaluate(np.array(X_test)[:,np.newaxis],y_test,verbose=0,batch_size=8)
score_train=model.evaluate(X_batch[:,np.newaxis],y_batch,verbose=0,batch_size=8)
print ('score_train: %.2f, score_test: %.2f' %(score_train,score_test))

print('-'*30)
print('Predicting masks on test data...')
print('-'*30)
#y_pred=model.predict(preprocess(X_test,Y_test,param_prep)[0])
#y_pred=model.predict(preprocess(X_train,Y_train,param_prep)[0])
#%%

tt='test'


if tt is 'train':
    n1=np.random.randint(len(X_train),size=100)
    X,Y=utils.preprocess_XY(X_train[n1],Y_train[n1],param_prep)
else:
    X,Y=X_test,y_test

# prediction
Y_pred=model.predict(X)>.5    

plt.figure(figsize=(20,20))
n1=utils.disp_img_2masks(np.expand_dims(X[:,3,:],axis=1),np.array(Y_pred,'uint8'),Y,4,4,0)
plt.show()
#%%
# deply on test data
#path2luna='./output/data/luna/test_luna_nodules_cin7.hdf5'
#ff_test=h5py.File(path2luna,'w-')
#
#print ss_test
#t1=h5py.File(ss_test[0],'r')
#for k in t1.keys():
#    print k
#    XY=t1[k]
#    X0=XY[0]
#    Y0=XY[1]
#    X1=[]
#    step=c_in
#    for k2 in range(0,X0.shape[0]-c_in,step):
#        X1.append(X0[k2:k2+c_in])
#    X1=np.stack(X1)
#    Y_pred=model.predict(utils.preprocess(X1,param_prep))>.5
#    ff_test[k]=Y_pred    
#ff_test.close()        


#Y_pred=utils.array_resize(Y_pred,(256,256))
#X0p,Y0p=utils.preprocess_XY(X0[::step,np.newaxis],Y0[::step,np.newaxis],param_prep)
#utils.disp_img_2masks(X1[1],Y_pred[1],None,4,5,0,range(0,20))

#%%     
X_batch,Y_batch=utils.preprocess_XY(X_batch,Y_batch,param_prep)
c1=1
X1=X_batch[:,c1,:]
Y1=Y_batch[:,c1,:]
X1=X1[:,np.newaxis,:]
Y1=Y1[:,np.newaxis,:]
plt.figure()
n1=utils.disp_img_2masks(X1,Y1,None,3,3,0,range(8))
plt.show()