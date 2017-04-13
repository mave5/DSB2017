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

#subset_list=glob(path2chunks+'subset*.hdf5')
#subset_list.sort()
#print 'total subsets: %s' %len(subset_list)

#%%

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
        rect = cv2.minAreaRect(largestcontour)
        rect=np.sort(rect[1])
        if rect[0]>0.:
            twod_ratio=rect[1]/rect[0]
        else:
            twod_ratio=3
        #print twod_ratio,radius
        
        #print x,y,radius
    else:
        #(x,y),radius=(np.random.randint(H,size=2)),20
        (x,y),radius=(0,0),0
        twod_ratio=3    
        #print x,y,radius
        #raise IOError
    return x,y,radius,twod_ratio

def mask2coord(Y):
    Y=np.array(Y>.5,dtype='uint8')
    if len(Y.shape)==2:
        Y=Y[np.newaxis,np.newaxis]
    elif len(Y.shape)==3:
        Y=Y[:,np.newaxis]

    N,C,H,W=Y.shape
        
    coords=np.zeros((N,4))
    for k in range(N):
        coords[k,:]=getRegionFromMap(Y[k,0])
    
    return coords
    
def resample(image, spacing, new_spacing=[1,1]):
    # Determine current pixel spacing
    #spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)
    
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    #new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image
    
def resample_vCord(voxelCoord, spacing, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    
    resize_factor = spacing / new_spacing
    new_voxelCoord = voxelCoord * resize_factor
    new_voxelCoord = np.round(new_voxelCoord).astype('int16')
    #real_resize_factor = new_shape / voxelCoord
    #new_spacing = spacing / real_resize_factor
    return new_voxelCoord
    
def collect_features(data_frame,dsb_type='train'):

    y_cancer = data_frame['cancer'].as_matrix()
    
    # load nodules    
    if dsb_type=='train':
        ff_node=h5py.File(path2dsb_nodes,'r')
        ff_dsb=h5py.File(path2dsb_resample,'r')
        ff_dsb_spacing=h5py.File(path2dsb_spacing,'r')
    elif dsb_type=='test':
        ff_node=h5py.File(path2dsbtest_nodes,'r')
        ff_dsb=h5py.File(path2dsbtest_resample,'r')
        ff_dsb_spacing=h5py.File(path2dsbtest_spacing,'r')
    elif dsb_type=='stage2':
        ff_node=h5py.File(path2dsbstage2_nodes,'r')
        ff_dsb=h5py.File(path2stage2_lung,'r')
        #ff_dsb_spacing=h5py.File(path2dsbtest_spacing,'r')
        
    
    extracted_features=[]
    # loop over all subjects    
    for k,id in enumerate(data_frame.id):
        print k,id,y_cancer[k]
        
        # read group features and labels    
        grp=ff_node[id]
        #print grp.keys()
        Y=grp['Y']#.value
        #cnz=grp['cnz'].value
        nzYi=grp['nzYi']#.value
        #sYi=grp['sYi']#.value
       
        # real lung
        #X_dsb=ff_dsb[id]['X']       
        Y_lung=ff_dsb[id]['Y']   
        #print Y_lung.shape
        
        # get spacing
        if dsb_type=='stage2':
            grp1=ff_dsb_stage2[id]                    
            spacing=grp1['spacing'].value        
        else:
            spacing=ff_dsb_spacing[id].value
        #print spacing
    
        # resample nodes
        Y_nz=[]
        z_locs=[]
        top_n=20
        for k2 in nzYi[0]:
            #print k2
            # resize nodes to 512*512
            Y_r = cv2.resize(Y[k2,0], (W, H), interpolation=cv2.INTER_CUBIC)                
            
            # convert original coords to equal spacing
            if dsb_type=='stage2':
                z_loc=k2*3+3
                Y_r=Y_lung[z_loc]*Y_r
                # resample nodes         
                Y_r=resample(np.array(Y_r>0.5,'uint8'),spacing[1:])>0.
            else:
                z_loc,_,_=resample_vCord([k2*3+3,H,W],spacing)
                # resample nodes         
                Y_r=resample(np.array(Y_r>0.5,'uint8'),spacing[1:])>0.
                Y_r=Y_lung[z_loc]*Y_r

            if np.sum(Y_r)>0:
                Y_nz.append(Y_r)
                z_locs.append(z_loc)               
                
        # collect some features
        #regs=[]
        #nb_regs=[]
        coords=[]
        #features=[]
        for ind in range(len(Y_nz)):
            xyr_2dratio=mask2coord(Y_nz[ind])
            coords.append(xyr_2dratio)
            #print ind, xyr
            #plt.figure()
            #plt.imshow(Y_nz[ind])
                    
        # convert to numpy
        if len(coords)>0:
            coords=np.vstack(coords)
            z_locs=np.vstack(z_locs)
            
            # pick nodules grater than xmm
            coords=coords[coords[:,2]>4]            

            # sort coords by node size            
            #print coords
            sort_ci=np.argsort(-coords[:,2])
            #print sort_ci
            # pick top node sizes
            coords=coords[sort_ci[:top_n]]
            coords=np.append(coords,np.zeros((top_n-coords.shape[0],coords.shape[1])),axis=0)
            #print coords.shape
            extracted_feature=np.reshape(coords,(1,coords.shape[0]*coords.shape[1]))
            #print extracted_feature
        else:    
            extracted_feature=np.zeros(top_n*4)
            #print extracted_feature
            
        extracted_features.append(extracted_feature)
        
    # stack all features
    x=np.vstack(extracted_features)
    print 'collected features shape:', x.shape
    
    return x
    
    
#%%

# nodule candiates of stage 1 train data
path2dsb_nodes='/media/mra/My Passport/Kaggle/datascience2017/hpz440/LUNA2016/hdf5/nodes/nfolds_dsb_nodes.hdf5'
if os.path.exists(path2dsb_nodes):
    ff_node=h5py.File(path2dsb_nodes,'r')
    #ff_r2=h5py.File(path2dsb_features2,'r')
    print 'dsb nodes:', len(ff_node)
else:
    print 'does not exist'

# nodule candiates of stage 1 test data
path2dsbtest_nodes='/media/mra/My Passport/Kaggle/datascience2017/hpz440/LUNA2016/hdf5/nodes/nfolds_dsbtest_nodes.hdf5'
if os.path.exists(path2dsbtest_nodes):
    ff_node=h5py.File(path2dsbtest_nodes,'r')
    #ff_r2=h5py.File(path2dsb_features2,'r')
    print 'dsb nodes:', len(ff_node)
else:
    print 'does not exist'

# nodule candiates of stage 2 test data
path2dsbstage2_nodes='/media/mra/My Passport/Kaggle/datascience2017/hpz440/LUNA2016/hdf5/nodes/nfolds_stage2_nodes.hdf5'
if os.path.exists(path2dsbstage2_nodes):
    ff_node=h5py.File(path2dsbstage2_nodes,'r')
    #ff_r2=h5py.File(path2dsb_features2,'r')
    print 'dsb nodes:', len(ff_node)
else:
    print 'does not exist'


# read dsb resample
path2dsb_resample=path2dsb_internal+'dsb_resampleXY.hdf5'
ff_dsb=h5py.File(path2dsb_resample,'r')
print 'stage1 train resample: ', len(ff_dsb)
step=24

# load metadata
path2dsb_spacing=path2dsb_external+'dsb_spacing.hdf5'
ff_dsb_spacing=h5py.File(path2dsb_spacing,'r')
print 'stage1 train spacing:', len(ff_dsb_spacing)

# path 2 dsb test resample
path2dsbtest_resample=path2dsb_internal+'dsbtest_resampleXY.hdf5'
ff_dsbtest=h5py.File(path2dsbtest_resample,'r')
print 'stage 1 test resample:', len(ff_dsbtest)

# load metadata for dsb test
path2dsbtest_spacing=path2dsb_external+'dsbtest_spacing.hdf5'
ff_dsbtest_spacing=h5py.File(path2dsbtest_spacing,'r')
print 'stage 1 test spacing:', len(ff_dsbtest_spacing)

# stage 2 lung masks
path2stage2_lung=path2dsb_internal+'stage2_lung.hdf5'
ff_stage2_lung=h5py.File(path2stage2_lung,'r')
print 'stage 2 lungs:', len(ff_stage2_lung)

# stage 2 data
path2stage2=path2dsb_internal+'stage2.hdf5'
ff_dsb_stage2=h5py.File(path2stage2,'r')
print 'stage 2 data:', len(ff_dsb_stage2)


#%%    

# load csv state 1 train ids
df_train = pd.read_csv('../stage1_labels.csv')
print('Number of training patients: {}'.format(len(df_train)))
print('Cancer rate: {:.4}%'.format(df_train.cancer.mean()*100))
df_train.head()
y_cancer = df_train['cancer'].as_matrix()


# load csv state 1 test ids
df_test = pd.read_csv('../stage1_solution.csv')
print('Number of test patients: {}'.format(len(df_test)))
df_test.head()
y_stage1 = df_test['cancer'].as_matrix()

# load csv state 1 test ids
df_stage2 = pd.read_csv('../stage2_sample_submission.csv')
print('Number of test patients: {}'.format(len(df_stage2)))
df_stage2.head()

#%%

path2features='./output/features/nodefeatures/nodefeatures_stage2.npz'

if not os.path.exists(path2features):
    # extract features
    X_dsb_train=collect_features(df_train,'train')    
    print X_dsb_train.shape
    X_dsb_stage1=collect_features(df_test,'test') 
    print X_dsb_stage1.shape
    X_dsb_stage2=collect_features(df_stage2,'stage2')    
    print X_dsb_stage2.shape
    
    # save features
    np.savez(path2features,X_dsb_train=X_dsb_train,X_dsb_stage1=X_dsb_stage1,X_dsb_stage2=X_dsb_stage2)
else:
    grp_features=np.load(path2features)
    X_dsb_train=grp_features['X_dsb_train']
    X_dsb_stage1=grp_features['X_dsb_stage1']
    X_dsb_stage2=grp_features['X_dsb_stage2']
    

#%%

# train XGBOOST

# concat dsb train and stage1
y_dsb_train = df_train['cancer'].as_matrix()
y_dsb_stage1 = df_test['cancer'].as_matrix()
y_dsb=np.append(y_dsb_train,y_dsb_stage1)
X_dsb=np.append(X_dsb_train,X_dsb_stage1,axis=0)


n_folds=5
# train validation split
skf = list(StratifiedKFold(y_dsb, n_folds))

# xgboost classifier
np.random.seed(10)
seed=np.random.randint(99999)
clf = xgb.XGBRegressor(max_depth=6,
                           n_estimators=3000,
                           min_child_weight=7,
                           learning_rate=0.03,
                           nthread=8,
                           subsample=0.80,
                           colsample_bytree=0.80,
                           seed=seed)

# train and validation
loss=[]
y_stage2_pred=[] #np.zeros(len(X_dsb_stage2))
for i, (train, test) in enumerate(skf):
            print "Fold", i
            x_train = X_dsb[train]
            y_train = y_dsb[train]
            x_test = X_dsb[test]
            y_test = y_dsb[test]
            clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=True, eval_metric='logloss', early_stopping_rounds=20)
            y_pred=clf.predict(x_test,ntree_limit=clf.best_iteration)
            loss.append(log_loss(y_test,y_pred))
            y_stage2_pred.append(clf.predict(X_dsb_stage2))

print 'average loss: %.3f' %(np.mean(loss))

# average
y_stage2_pred=np.array(y_stage2_pred)
print y_stage2_pred.shape
y_stage2_pred=np.mean(y_stage2_pred,axis=0)
print y_stage2_pred.shape
# plot feature importance
#plot_importance(clf)
#plt.show()


#%%
df = pd.read_csv('../stage2_sample_submission.csv')
print('Number of training patients: {}'.format(len(df)))
df.head()

df['cancer'] = y_stage2_pred
    
import datetime
now = datetime.datetime.now()
info='loss_'+str(np.mean(loss))
suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
sub_file = os.path.join('./output/submission', 'submission_' + suffix + '.csv')

df.to_csv(sub_file, index=False)
print(df.head())    


