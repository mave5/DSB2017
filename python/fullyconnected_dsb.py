#%% deploy classify nodes on DSB
import numpy as np
import cv2
import time
import os
import matplotlib.pylab as plt
import pandas as pd
import models
import utils
from keras import backend as K
from keras.utils import np_utils
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
path2dsb_features=path2dsb_internal+'dsb_features2.hdf5'
ff_dsb_features=h5py.File(path2dsb_features,'r')

df_train = pd.read_csv('../stage1_labels.csv')
print('Number of training patients: {}'.format(len(df_train)))
print('Cancer rate: {:.4}%'.format(df_train.cancer.mean()*100))
df_train.head()

x=[]
start_point=700
end_point=1397

# path to csv file to save scores
path2scorescsv = './output/features/features.csv'
first_row = 'id,test'
with open(path2scorescsv, 'w+') as f:
    f.write(first_row + '\n')


# labels
y = df_train['cancer'].as_matrix()
y=y[start_point:end_point]

for k,id in enumerate(df_train.id[start_point:end_point]):
    print k,id,y[k]
    grp=ff_dsb_features[id]

    feature=grp['features'].value
    # max pool over 2*2*2
    feature=np.max(feature,axis=(2,3,4))
    # max feature
    #feature=np.max(feature,axis=0)
    
    out=grp['output']
    y_pred=np.argmax(grp['output'],axis=1)    
    sort_i=np.argsort(y_pred)        
    
    x.append(feature[sort_i[-1]])
    #print 'std:', np.std(feature[y_pred>0]),feature[y_pred>0].shape
    

asdasdas

x=np.stack(x)


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
#x1=np.zeros((256,512))
#y1=np.zeros((256,1))
#x2=np.ones((256,512))
#y2=np.ones((256,1))
#x3=np.concatenate((x1,x2))
#y3=np.concatenate((y1,y2))

params={
        'h0':512,
        'h1':100,
        'h2':10,
        'learning_rate': 3e-4,
        'loss': 'binary_crossentropy',
        'num_labels': 1,
        }

model=models.model_fc(params)
model.summary()

#model.fit(x3,y3,validation_data=(x3,y3),nb_epoch=30,shuffle=True,batch_size=64,verbose=1)
best_loss=100
for kk in range(100):
    model.fit(trn_x,trn_y,validation_data=(val_x,val_y),nb_epoch=1,shuffle=True,batch_size=8,verbose=0)
    loss_test=model.evaluate(val_x,val_y,verbose=0)
    if loss_test<best_loss:
        print loss_test
        best_loss=loss_test
    


#%%

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


