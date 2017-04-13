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
from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedKFold

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



def getRegionFromMap(Y):
    Y=np.array(Y>0.5,'uint8')
    im2, contours, hierarchy = cv2.findContours(Y,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    
    areaArray=[]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaArray.append(area)
    # check for any contour    
    if len(areaArray)>0:    
        largestcontour=contours[np.argmax(areaArray)]
        (x,y),radius = cv2.minEnclosingCircle(largestcontour)
        #print x,y,radius
    else:
        #(x,y),radius=(np.random.randint(H,size=2)),20
        (x,y),radius=(0,0),0
        #print x,y,radius
        #raise IOError
    return x,y,radius

def mask2coord(Y):
    Y=np.array(Y>.5,dtype='uint8')
    if len(Y.shape)==2:
        Y=Y[np.newaxis,np.newaxis]
    elif len(Y.shape)==3:
        Y=Y[:,np.newaxis]

    N,C,H,W=Y.shape
        
    coords=np.zeros((N,3))
    for k in range(N):
        coords[k,:]=getRegionFromMap(Y[k,0])
    
    return coords
    

#%%

path2dsbnodes='/media/mra/My Passport/Kaggle/datascience2017/hpz440/LUNA2016/hdf5/nodes/nfolds_dsb_nodes.hdf5'
if os.path.exists(path2dsbnodes):
    ff_r=h5py.File(path2dsbnodes,'r')
    #ff_r2=h5py.File(path2dsb_features2,'r')
    print 'dsb nodes:', len(ff_r)
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


#%%
nodesize=[]
y_cancer=y
extracted_features=[]

for k,id in enumerate(df_train.id):
    print k,id,y[k]
    
    # read group features and labels    
    grp=ff_r[id]
    #print grp.keys()
    Y=grp['Y']#.value
    cnz=grp['cnz'].value
    nzYi=grp['nzYi']#.value
    sYi=grp['sYi']#.value
   
    # collect some features
    p1=[]
    top_nodesize=[]
    #top_n=10
    regs=[]
    nb_regs=[]
    coords=[]
    features=[]
    for ind in nzYi[0]:
        xyr=mask2coord(Y[ind])
        coords.append(xyr)
        #print ind, xyr

    # convert to numpy
    coords=np.vstack(coords)
    
    # clustering
    num_regs=6
    if len(coords)>num_regs:
        kmeans_model = KMeans(n_clusters=num_regs, random_state=1).fit(coords[:,:2])
        labels = kmeans_model.labels_
    else:
        labels=np.zeros(len(coords))
        
    
    # number of node in each corner    
    extracted_feature=[]
    coords_by_region=[]
    for k1 in range(num_regs):
        # coordinates by region
        coords_by_region.append(coords[labels==k1]) 
        # extracted features
        extracted_feature.append(len(coords_by_region[k1]))

    
    # average node size each side
    for k1 in range(num_regs):
        if len(coords[labels==k1])>0:
            # stats of diameter
            tmp1=np.mean(coords[labels==k1,2])
            tmp2=np.max(coords[labels==k1,2])
            tmp3=np.std(coords[labels==k1,2])

            # stats of x
            tmp4=np.mean(coords[labels==k1,0])
            tmp5=np.max(coords[labels==k1,0])
            tmp6=np.std(coords[labels==k1,0])

            # stats of y
            tmp7=np.mean(coords[labels==k1,1])
            tmp8=np.max(coords[labels==k1,1])
            tmp9=np.std(coords[labels==k1,1])
            
        else:
            tmp1=0
            tmp2=0
            tmp3=0

            tmp4=0
            tmp5=0
            tmp6=0

            tmp7=0
            tmp8=0
            tmp9=0

            
        extracted_feature.append(tmp1)        
        extracted_feature.append(tmp2)        
        extracted_feature.append(tmp3)        
        extracted_feature.append(tmp4)        
        extracted_feature.append(tmp5)                
        extracted_feature.append(tmp6)                
        
        extracted_feature.append(tmp7)        
        extracted_feature.append(tmp8)                
        extracted_feature.append(tmp9)                
        
    #print extracted_feature
    extracted_features.append(extracted_feature)

# stack all features
x=np.stack(extracted_features)


#%%

#%%

# test data

df_test = pd.read_csv('../stage1_submission.csv')
print('Number of test patients: {}'.format(len(df_test)))
df_test.head()


path2dsbnodes='/media/mra/My Passport/Kaggle/datascience2017/hpz440/LUNA2016/hdf5/nodes/nfolds_dsbtest_nodes.hdf5'
if os.path.exists(path2dsbnodes):
    ff_r=h5py.File(path2dsbnodes,'r')
    #ff_r2=h5py.File(path2dsb_features2,'r')
    print 'dsb nodes:', len(ff_r)
else:
    print 'does not exist'

#%%

extracted_features=[]
for k,id in enumerate(df_test.id):
    print k,id
    
    # read group features and labels    
    grp=ff_r[id]
    #print grp.keys()
    Y=grp['Y']#.value
    cnz=grp['cnz'].value
    nzYi=grp['nzYi']#.value
    sYi=grp['sYi']#.value
       
    # collect some features
    p1=[]
    top_nodesize=[]
    #top_n=10
    regs=[]
    nb_regs=[]
    coords=[]
    features=[]
    for ind in nzYi[0]:
        xyr=mask2coord(Y[ind])
        coords.append(xyr)
        #print ind, xyr
    
    # plot coords
    coords=np.vstack(coords)
    
    # clustering
    num_regs=6
    if len(coords)>num_regs:
        kmeans_model = KMeans(n_clusters=num_regs, random_state=1).fit(coords[:,:2])
        labels = kmeans_model.labels_
    else:
        labels=np.zeros(len(coords))
        
    
    # number of node in each corner    
    extracted_feature=[]
    coords_by_region=[]
    for k1 in range(num_regs):
        # coordinates by region
        coords_by_region.append(coords[labels==k1]) 
        # extracted features
        extracted_feature.append(len(coords_by_region[k1]))

    
    # average node size each side
    for k1 in range(num_regs):
        if len(coords[labels==k1])>0:
            # stats of diameter
            tmp1=np.mean(coords[labels==k1,2])
            tmp2=np.max(coords[labels==k1,2])
            tmp3=np.std(coords[labels==k1,2])

            # stats of x
            tmp4=np.mean(coords[labels==k1,0])
            tmp5=np.max(coords[labels==k1,0])
            tmp6=np.std(coords[labels==k1,0])

            # stats of y
            tmp7=np.mean(coords[labels==k1,1])
            tmp8=np.max(coords[labels==k1,1])
            tmp9=np.std(coords[labels==k1,1])
            
        else:
            tmp1=0
            tmp2=0
            tmp3=0

            tmp4=0
            tmp5=0
            tmp6=0

            tmp7=0
            tmp8=0
            tmp9=0

            
        extracted_feature.append(tmp1)        
        extracted_feature.append(tmp2)        
        extracted_feature.append(tmp3)        
        extracted_feature.append(tmp4)        
        extracted_feature.append(tmp5)                
        extracted_feature.append(tmp6)                
        
        extracted_feature.append(tmp7)        
        extracted_feature.append(tmp8)                
        extracted_feature.append(tmp9)                
        
    #print extracted_feature
    extracted_features.append(extracted_feature)
    
# stack features
x_dsb_test=np.stack(extracted_features)
print x_dsb_test.shape

#%%

y = df_train['cancer'].as_matrix()
n_folds=5
skf = list(StratifiedKFold(y, n_folds))

clf = xgb.XGBRegressor(max_depth=3,
                           n_estimators=70,
                           min_child_weight=6,
                           learning_rate=0.07,
                           nthread=8,
                           subsample=0.70,
                           colsample_bytree=0.80,
                           seed=4241)

loss=[]
for i, (train, test) in enumerate(skf):
            print "Fold", i
            x_train = x[train]
            y_train = y[train]
            x_test = x[test]
            y_test = y[test]
            clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=True, eval_metric='logloss', early_stopping_rounds=5)
            y_pred=clf.predict(x_test)
            loss.append(log_loss(y_test,y_pred))

print 'average loss: %.3f' %(np.mean(loss))

#%%
y_dsb_pred=np.zeros(len(x_dsb_test))
for i, (train, test) in enumerate(skf):
            print "Fold", i
            x_train = x[train]
            y_train = y[train]
            x_test = x[test]
            y_test = y[test]
            clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=True, eval_metric='logloss', early_stopping_rounds=5)
            y_dsb_pred+=clf.predict(x_dsb_test)
# average
y_dsb_pred=y_dsb_pred/n_folds



#%%
df = pd.read_csv('../stage1_submission.csv')
print('Number of training patients: {}'.format(len(df)))
df.head()

df['cancer'] = y_dsb_pred
    
import datetime
now = datetime.datetime.now()
info='nfolds'
suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
sub_file = os.path.join('./output/submission', 'submission_' + suffix + '.csv')

df.to_csv(sub_file, index=False)
print(df.head())    


