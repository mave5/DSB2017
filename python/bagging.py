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
#from glob import glob
#from sklearn import cross_validation
import xgboost as xgb
#from mpl_toolkits.mplot3d import Axes3D
#from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd    
from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedKFold
import scipy.ndimage
from xgboost import plot_importance
#from keras.utils import np_utils
#from skimage import measure
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
#from keras.models import load_model
#%%
# settings
H,W=512,512

# path to dataset
path2luna_external="/media/mra/My Passport/Kaggle/datascience2017/hpz440/LUNA2016/hdf5/"
#path2subsets=path2luna_external+"subsets/"
#path2chunks=path2luna_external+"chunks/"
#path2luna_internal="/media/mra/win71/data/misc/kaggle/datascience2017/LUNA2016/"
#path2chunks=path2luna_internal+"chunks/"

path2dsb_external='/media/mra/My Passport/Kaggle/datascience2017/hpz440/data/hdf5/'
path2dsb_internal="/media/mra/win71/data/misc/kaggle/datascience2017/data/"



#%%    

# load csv state 1 train ids
df_train = pd.read_csv('../stage1_labels.csv')
print('Number of training patients: {}'.format(len(df_train)))
print('Cancer rate: {:.4}%'.format(df_train.cancer.mean()*100))
df_train.head()
y_cancer = df_train['cancer'].as_matrix()


# load csv state 1 test ids
df_test = pd.read_csv('../stage1_submission.csv')
print('Number of test patients: {}'.format(len(df_test)))
df_test.head()

#%%

# load features for future use
path2dsb_features='./output/features/nodefeatures/nodefeatures.npz'
if os.path.exists(path2dsb_features): 
    grp=np.load(path2dsb_features)
    X_dsb=grp['X_dsb']
    X_dsb_test=grp['X_dsb_test']    
    y_dsb=grp['y_dsb']
    print 'features loaded!'
else:
    print 'features does not exist!'

#%%


from brew.base import Ensemble, EnsembleClassifier
from brew.combination.combiner import Combiner

# train XGBOOST
y_dsb = df_train['cancer'].as_matrix()
n_folds=10

# train validation split
skf = list(StratifiedKFold(y_dsb, n_folds))

# xgboost classifier
clf1 = xgb.XGBRegressor(max_depth=4,
                           n_estimators=100,
                           min_child_weight=7,
                           learning_rate=0.06,
                           nthread=8,
                           subsample=0.70,
                           colsample_bytree=0.80,
                           seed=4241)

# xgboost classifier
clf2 = xgb.XGBRegressor(max_depth=3,
                           n_estimators=150,
                           min_child_weight=6,
                           learning_rate=0.06,
                           nthread=8,
                           subsample=0.50,
                           colsample_bytree=0.90,
                           seed=1)

# xgboost classifier
seed=np.random.seed(0)
clf3 = xgb.XGBRegressor(max_depth=5,
                           n_estimators=350,
                           min_child_weight=4,
                           learning_rate=0.06,
                           nthread=8,
                           subsample=0.60,
                           colsample_bytree=0.70,
                           seed=seed)


# Creating Ensemble
ensemble = Ensemble([clf1, clf2, clf3])
eclf = EnsembleClassifier(ensemble=ensemble, combiner=Combiner('mean'))

clf_list = [clf1, clf2, clf3]
lbl_list = ['xgboost 1', 'xgboost 2', 'xgboost 3', 'Ensemble']


loss=[]
y_pred_dsbtest=[]
for clf, lab in zip(clf_list, lbl_list):
    print lab
    
    # train and validation
    #loss=[]
    y_dsb_pred=np.zeros(len(X_dsb_test))
    for i, (train, test) in enumerate(skf[2:]):
                print "Fold", i
                x_train = X_dsb[train]
                y_train = y_dsb[train]
                x_test = X_dsb[test]
                y_test = y_dsb[test]
                clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False, eval_metric='logloss', early_stopping_rounds=20)
                y_pred=clf.predict(x_test,ntree_limit=clf.best_iteration)
                loss.append(log_loss(y_test,y_pred))
                y_pred=clf1.predict(X_dsb_test)
                y_pred_dsbtest.append(y_pred)

print 'average loss: %.3f' %(np.mean(loss))
#%%

df = pd.read_csv('../stage1_submission.csv')
print('Number of training patients: {}'.format(len(df)))
df.head()

# average of ensemble
y_pred_test=np.array(y_pred_dsbtest)
y_pred_test=np.mean(y_pred_test,axis=0)

df['cancer'] = y_pred_test
    
import datetime
now = datetime.datetime.now()
info='loss_'+str(np.mean(loss))
suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
sub_file = os.path.join('./output/submission', 'submission_' + suffix + '.csv')

df.to_csv(sub_file, index=False)
print(df.head())    


