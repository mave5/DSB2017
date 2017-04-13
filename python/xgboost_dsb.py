#%% deploy classify nodes on DSB
import numpy as np
import cv2
import time
import os
import matplotlib.pylab as plt
import pandas as pd
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

#%%

# fold number
foldnm=1

# seed point
seed = 2017
seed = np.random.randint(seed)


#%%

# deploy on DSB data
path2dsb_features=path2dsb_internal+'dsb_features.hdf5'
ff_dsb_features=h5py.File(path2dsb_features,'r')

df_train = pd.read_csv('../stage1_labels.csv')
print('Number of training patients: {}'.format(len(df_train)))
print('Cancer rate: {:.4}%'.format(df_train.cancer.mean()*100))
df_train.head()

x=[]
for id in df_train.id:
    print id
    tmp=ff_dsb_features[id]
    
    x.append(tmp)
y = df_train['cancer'].as_matrix()

x=np.stack(x)
#x=np.reshape(x,(x.shape[0],256,2,2,2))
#x=np.max(x,axis=(2,3,4))

# extract non cancers
non_cancer=df_train[df_train.cancer==0].id
cancer=df_train[df_train.cancer==1].id
print 'total non cancer:%s, total cancer:%s' %(len(non_cancer),len(cancer))



x_c=x[y==1]
x_nc=x[y==0]

y_nc=np.zeros(len(x_nc),'uint8')
y_c=np.ones(len(x_c),'uint8')

# indices of train and validation
xt=np.concatenate((x_c,x_nc[:len(x_c)]))
yt=np.concatenate((y_c,y_nc[:len(y_c)]))

#%%
trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=42, stratify=y,
                                                               test_size=0.20)


clf = xgb.XGBRegressor(max_depth=10,
                           n_estimators=1500,
                           min_child_weight=2,
                           learning_rate=0.07,
                           nthread=8,
                           subsample=0.80,
                           colsample_bytree=0.80,
                           seed=4242)

clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=50)


#%%
dasda
# deploy on DSB data
path2dsbtest_features=path2dsb_internal+'dsbtest_features.hdf5'
ff_dsbtest_features=h5py.File(path2dsbtest_features,'r')

df = pd.read_csv('../stage1_submission.csv')
print('Number of training patients: {}'.format(len(df)))
df.head()

x=[]
for id in df.id:
    print id
    tmp=ff_dsbtest_features[id]
    
    x.append(tmp)

x=np.stack(x)
x=np.reshape(x,(x.shape[0],256,2,2,2))
x=np.max(x,axis=(2,3,4))

y_pred=clf.predict(x)

df['cancer'] = y_pred
    
import datetime
now = datetime.datetime.now()
info='rm'
suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
sub_file = os.path.join('./output/submission', 'submission_' + suffix + '.csv')

df.to_csv(sub_file, index=False)
print(df.head())    


