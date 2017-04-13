#%% deploy classify nodes on DSB
import numpy as np
import cv2
import time
import os
import matplotlib.pylab as plt
#import models
#import utils
#from keras import backend as K
#from keras.utils import np_utils
import h5py
from glob import glob
from sklearn import cross_validation
import xgboost as xgb
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd    
#from keras.utils import np_utils
#from skimage import measure
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
#from keras.models import load_model
#%%
# settings

# path to dataset
path2luna_external="/media/mra/My Passport/Kaggle/datascience2017/hpz440/LUNA2016/hdf5/"
path2subsets=path2luna_external+"subsets/"
path2chunks=path2luna_external+"chunks/"
#path2luna_internal="/media/mra/win71/data/misc/kaggle/datascience2017/LUNA2016/"
#path2chunks=path2luna_internal+"chunks/"

path2dsb_external='/media/mra/My Passport/Kaggle/datascience2017/hpz440/data/hdf5/'
path2dsb_internal="/media/mra/win71/data/misc/kaggle/datascience2017/data/"

#subset_list=glob(path2chunks+'subset*.hdf5')
#subset_list.sort()
#print 'total subsets: %s' %len(subset_list)

#%%

# conver chunk number to region
#def chunknb2zyx(chunk_nb,(N,H,W,step)):
#    n=np.ceil(N*1./step)
#    h=np.ceil(H*1./step)
#    w=np.ceil(W*1./step)
#    nb_chunks=n*h*w
#    if chunk_nb>nb_chunks:
#        raise IOError
#    z,r=divmod(chunk_nb,(h*w))
#    y,x=divmod(r,(h))
#        
#    if (x<=w/2) & (y<=w/2):
#        loc=0 #'top left'
#    elif (x>w/2) & (y<=w/2):
#        loc=1 #'top right'
#    elif (x<=w/2) & (y>w/2):
#        loc=2 #'bottom left'        
#    elif (x>w/2) & (y>w/2):
#        loc=3 #'bottom right'        
#    return (z*step,y*step,x*step)
    
    
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
    
def extract_chunk(X,w,chunk_nb):
    # chunks
    N,H,W=X.shape    
    step=24
    
    cz,cy,cx=chunknb2zyx(chunk_nb,(N,H,W,step))
    #print cz,cy,cx
    
    # extract over lapping chunks
    cz=int(cz*step)
    cx=int((cx)*step)
    cy=int((cy)*step)
    print cx,cy,cz
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
    

#%%

path2dsb_features=path2dsb_internal+'dsb_features.hdf5'
#path2dsb_features=path2dsb_external+'dsb_features.hdf5'
#path2dsb_features2=path2dsb_internal+'dsb_features2.hdf5'
if os.path.exists(path2dsb_features):
    ff_r=h5py.File(path2dsb_features,'r')
    #ff_r2=h5py.File(path2dsb_features2,'r')
    print len(ff_r)
    #print len(ff_r2)
else:
    print 'does not exist'



# read dsb resample
path2dsb_resample=path2dsb_internal+'dsb_resampleXY.hdf5'
ff_dsb=h5py.File(path2dsb_resample,'r')
step=24


#%%    

df_train = pd.read_csv('../stage1_labels.csv')
print('Number of training patients: {}'.format(len(df_train)))
print('Cancer rate: {:.4}%'.format(df_train.cancer.mean()*100))
df_train.head()
y = df_train['cancer'].as_matrix()


# combine two hdf5 files
#path2dsb_combine=path2dsb_internal+'dsb_features.hdf5'
#ff_r12=h5py.File(path2dsb_combine,'w-')
#
## combine two features
#for k,key in enumerate(df_train.id[:700]):
#    print k,key
#    grp=ff_r12.create_group(key)    
#    grp_temp=ff_r1[key]
#    grp['features']=grp_temp['features'].value
#    grp['output']=grp_temp['output'].value
#
#for k,key in enumerate(df_train.id[700:]):
#    print k,key
#    grp=ff_r12.create_group(key)    
#    grp_temp=ff_r2[key]
#    grp['features']=grp_temp['features'].value
#    grp['output']=grp_temp['output'].value
#
#
#ff_r12.close()
#


#%%
node_size=[]
l_label=[]
prob=[]
max_loc=[]
nodesize=[]
locs=[]
start_point=0
end_point=1397
y_cancer=y[start_point:end_point]
xn=[]

print 'wait to extract features ....'
feann=[]
pca = PCA(n_components=4)


for k,id in enumerate(df_train.id[start_point:end_point]):

    # read group features and labels    
    grp=ff_r[id]
    feature=grp['features']#.value
    # max pool over 2*2*2
    feature=np.max(feature,axis=(2,3,4))
    
    # get output
    y_pred=np.argmax(grp['output'],axis=1)
    sort_i=np.argsort(y_pred)        
    nz_i=np.nonzero(y_pred)
    #out=grp['output'].value
    #nz_y_pred=y_pred[nz_i]
    #prob=[]
    #for o1,out1 in enumerate(out[nz_i]):
        #print nz_y_pred[o1]
        #prob.append(out1[y_pred[o1]])

    #print k,id,y_cancer[k],np.mean(prob),np.max(prob)
   
    # original data shape    
    N,H,W=ff_dsb[id]['X'].shape
    #X=ff_dsb[id]['X']
    #Y=ff_dsb[id]['Y']
    
    # collect some features
    p1=[]
    top_nodesize=[]
    top_n=2
    regs=[]
    nb_regs=[]
    coords=[]
    features=[]
    for ind in nz_i[0]:
        node_size=y_pred[ind]
        top_nodesize.append(node_size)
        #features.append(feature[ind])
        
        # get location of each chunk
        zyx=chunknb2zyx(ind,(N,H,W,step))
        zyx=np.array(zyx)*step
        #print reg,zyx,node_size
        #regs.append(reg)
        coords.append(zyx)

    # plot coords
    coords=np.array(coords)
    top_nodesize=np.array(top_nodesize)
    features=np.array(features)
    
    # clustering
    num_regs=2
    if len(coords)>num_regs:
        kmeans_model = KMeans(n_clusters=num_regs, random_state=1).fit(coords)
        labels = kmeans_model.labels_
        
    else:
        labels=np.zeros(len(coords))

    minX=np.min(coords[:,2])    
    maxX=np.max(coords[:,2])    
    midX=(minX+maxX)/2.
    labels-coords[:,2]>midX
    
#    coords_by_region=[]
#    for k1 in range(num_regs):
#        coords_by_region.append(coords[labels==k1]) 
#
#    # collecting features
#    fea1=N
#    fea2=H
#    fea3=W

#    if len(features[labels==0]):     
#        cnn_features1=np.max(features[labels==0],axis=0)
#    else:
#        cnn_features1=np.zeros(feature.shape[1])        
#    if len(features[labels==1]):     
#        cnn_features2=np.max(features[labels==1],axis=0)
#    else:
#        cnn_features2=np.zeros(feature.shape[1])        
    
    # distance to center
#    if len(coords[labels==0])>0:
#        d0=np.sum(np.abs(coords[labels==0]-kmeans_model.cluster_centers_[0]))
#    else:
#        d0=0
#    if len(coords[labels==1])>0:        
#        d1=np.sum(np.abs(coords[labels==1]-kmeans_model.cluster_centers_[1]))
#    else:
#        d1=0
#    d0=d0/(d0+d1+1)        
#    d1=d1/(d0+d1+1)        
        
    # number of nodes each side
#    if len(coords)>0:        
#        fea4=1.*len(coords_by_region[0])#/len(coords)
#        fea5=1.*len(coords_by_region[1])#/len(coords)
#    else:
#        fea4=0
#        fea5=0
#    
#    # max node size each side
#    if len(top_nodesize[labels==0])>0:
#        fea6=np.sum(top_nodesize[labels==0])
#    else:
#        fea6=0
#    if len(top_nodesize[labels==1])>0:    
#        fea7=np.sum(top_nodesize[labels==1])
#    else:
#        fea7=0
        
    # min and max node X,Y,Z
#    if len(coords)>0:        
#        fea8=1.*np.min(coords[:,0])/N
#        fea9=1.*np.max(coords[:,0])/N
#    
#        fea10=1.*np.min(coords[:,1])/H
#        fea11=1.*np.max(coords[:,1])/H
#
#        fea12=1.*np.min(coords[:,2])/W
#        fea13=1.*np.max(coords[:,2])/W
#    else:
#        fea8=0.
#        fea9=0.
#    
#        fea10=0.
#        fea11=0.
#
#        fea12=0.
#        fea13=0.
        
    #feann.append(coords)   
    #feann.append([fea4,fea5,fea6,fea7,fea8,fea9,fea10,fea11,fea12,fea13])
    #feann.append([fea4,fea5,fea8,fea8,fea9,fea10,fea11,fea12,fea13,d0,d1])
    #feann.append([cnn_features1,cnn_features2])
     
    # pca    
    pca.fit(feature)
    #print(pca.explained_variance_ratio_)
     
    #feautreMax=feature[sort_i[-1]]
    feautreMax=pca.explained_variance_ratio_
    max_node_size=y_pred[sort_i[-1]]    
    feautreMax=np.append(feautreMax,max_node_size)
    feann.append(feautreMax)
        


x=np.stack(feann)
#x=np.reshape(x,(x.shape[0],x.shape[1]*x.shape[2]))
#%%
y = df_train['cancer'].as_matrix()

from sklearn.model_selection import train_test_split


trn_x, val_x, trn_y, val_y = train_test_split(x, y, random_state=42,  stratify=y,
                                                               test_size=0.20)

clf = xgb.XGBRegressor(max_depth=2,
                           n_estimators=500,
                           min_child_weight=9,
                           learning_rate=0.07,
                           nthread=8,
                           subsample=0.80,
                           colsample_bytree=0.80,
                           seed=4242)

clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=50)

#%%



