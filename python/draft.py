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
#from image import ImageDataGenerator
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

path2dsb_internal="/media/mra/win71/data/misc/kaggle/datascience2017/data/"

subset_list=glob(path2chunks+'subset*.hdf5')
subset_list.sort()
print 'total subsets: %s' %len(subset_list)

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
        
    if (x<=w/2) & (y<=w/2):
        loc=0 #'top left'
    elif (x>w/2) & (y<=w/2):
        loc=1 #'top right'
    elif (x<=w/2) & (y>w/2):
        loc=2 #'bottom left'        
    elif (x>w/2) & (y>w/2):
        loc=3 #'bottom right'        
    return loc,(z,y,x)

#%%

path2dsb_features=path2dsb_internal+'dsb_features2.hdf5'
if os.path.exists(path2dsb_features):
    ff_r=h5py.File(path2dsb_features,'r')
    print len(ff_r)
else:
    print 'does not exist'

# read dsb resample
path2dsb_resample=path2dsb_internal+'dsb_resampleXY.hdf5'
ff_dsb=h5py.File(path2dsb_resample,'r')
step=24

#%%    
import pandas as pd    
end_point=1397
df_train = pd.read_csv('../stage1_labels.csv')
print('Number of training patients: {}'.format(len(df_train)))
print('Cancer rate: {:.4}%'.format(df_train.cancer.mean()*100))
df_train.head()
y = df_train['cancer'][:end_point].as_matrix()

# save features
df_features = pd.read_csv('./output/features/stage1_labels_features.csv')
df_features.head()


node_size=[]
l_label=[]
prob=[]
max_loc=[]
nodesize=[]
locs=[]
start_point=706
end_point=707
y_cancer=y[start_point:end_point]

cnz=[]
cz=[]
c1s=[]
c2s=[]
c3s=[]
c4s=[]
c5s=[]
for k,id in enumerate(df_train.id[start_point:end_point]):
    print id,y_cancer[k]

    # read group features and labels    
    grp=ff_r[id]
    feature=grp['features']#.value
    # max pool over 2*2*2
    feature=np.max(feature,axis=(2,3,4))
    
    # get output
    y_pred=np.argmax(grp['output'],axis=1)
    sort_i=np.argsort(y_pred)        
    
    # count non zeros
    cnz.append(np.count_nonzero(y_pred))
    cz.append(np.count_nonzero(y_pred==0))
    c1s.append(np.count_nonzero(y_pred==1))    
    c2s.append(np.count_nonzero(y_pred==2))
    c3s.append(np.count_nonzero(y_pred==3))    
    c4s.append(np.count_nonzero(y_pred==4))    
    c5s.append(np.count_nonzero(y_pred==5))    
    
    
    # max feature
    #feature=np.max(feature,axis=1)
    #print np.max(feature)
    #plt.plot(feature)

    #print np.count_nonzero(y_pred==1),np.count_nonzero(y_pred==2),np.count_nonzero(y_pred==3),np.count_nonzero(y_pred==4)
        
    
    #y_pred=np.sort(y_pred) 
    
    # max node size    
    #mnz=y_pred[-5:]
    
    # probability
    p1=[]
    top_nodesize=[]
    top_n=2
    regs=[]
    nb_regs=[]
    coords=[]
    for ind in sort_i[-top_n:]:
        node_size=y_pred[ind]
        top_nodesize.append(node_size)
        p1.append(feature[ind])
        N,H,W=ff_dsb[id]['X'].shape
        reg,zyx=chunknb2zyx(ind,(N,H,W,step))
        print reg,zyx,node_size
        regs.append(reg)
        coords.append(zyx)
    regs=np.array(regs)        
    coords=np.array(coords)        
#
#    unique, ind,counts = np.unique(regs, return_counts=True, return_index=True)
#    dict(zip(unique, counts))
    

    #print top_nodesize,p1
    #mnz=y_pred[sort_i[-1]-top_n:sort_i[-1]+top_n+1]
    #nodesize.append(mnz)
    #locs.append(loc2)
    #nb_regs.append(nb_reg)
    
    #print 'cancer: %s' %(y_cancer[k])
    #print np.sum(mnz),np.sum(p1)
    prob.append(p1)
    


#%%
y = df_train['cancer'][:end_point].as_matrix()
x1=np.array(nodesize)
x2=np.array(prob)
x3=np.array(locs)
x=np.concatenate((x1,x2,x3),axis=1)
x=np.array(nb_regs)

trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=4242, stratify=y,
                                                               test_size=0.20)

clf = xgb.XGBRegressor(max_depth=10,
                           n_estimators=1500,
                           min_child_weight=2,
                           learning_rate=0.08,
                           nthread=8,
                           subsample=0.80,
                           colsample_bytree=0.80,
                           seed=4242)

clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=50)


    
    