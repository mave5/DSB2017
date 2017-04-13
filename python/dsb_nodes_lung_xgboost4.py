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
    else:
        ff_node=h5py.File(path2dsbtest_nodes,'r')
        ff_dsb=h5py.File(path2dsbtest_resample,'r')
        ff_dsb_spacing=h5py.File(path2dsbtest_spacing,'r')
    
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
        spacing=ff_dsb_spacing[id].value
        #print spacing
    
        # resample nodes
        Y_nz=[]
        z_locs=[]
        for k2 in nzYi[0]:
            #print k2
            # resize nodes to 512*512
            Y_r = cv2.resize(Y[k2,0], (W, H), interpolation=cv2.INTER_CUBIC)                
            # resample nodes         
            Y_r=resample(np.array(Y_r>0.5,'uint8'),spacing[1:])>0.
            
            # convert original coords to equal spacing
            z_loc,_,_=resample_vCord([k2*3+3,H,W],spacing)
            #plt.figure()                
            #plt.subplot(1,2,1)        
            #plt.imshow(Y_r)
            Y_r=Y_lung[z_loc]*Y_r
            if np.sum(Y_r)>0:
                Y_nz.append(Y_r)
                z_locs.append(z_loc)               
            #plt.subplot(1,2,2)        
            #plt.imshow(Y_r)
            
            
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
            #z_locs=z_locs[coords[:,2]>4]
            #coords_z=np.append(coords,z_locs,axis=1)            
            #print coords#,coords_z[:,(0,1,3)]

            # clustering
            num_regs=4
            if len(coords)>=num_regs:
                kmeans_model = KMeans(n_clusters=num_regs, random_state=1).fit(coords[:,:2])
                #kmeans_model = KMeans(n_clusters=num_regs, random_state=1).fit(coords_z[:,(0,1,3)])
                labels = kmeans_model.labels_
            else:
                #print len(coords)
                for n_reg in range(num_regs-1,0,-1):
                    if len(coords)>=n_reg:
                        kmeans_model = KMeans(n_clusters=n_reg, random_state=1).fit(coords[:,:2])
                        #kmeans_model = KMeans(n_clusters=num_regs, random_state=1).fit(coords_z[:,(0,1,3)])
                        labels = kmeans_model.labels_
                        #print labels
                        break
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
                #print coords[labels==k1],coords_z[labels==k1]
            
            # average node size each side
            for k1 in range(num_regs):
                if len(coords[labels==k1])>0:
                    # stats of node size 
                    tmp1=np.mean(coords[labels==k1,2])
                    tmp2=np.max(coords[labels==k1,2])
                    tmp3=np.median(coords[labels==k1,2])                                             
                    tmp4=np.std(coords[labels==k1,2])
                    #tmp4=np.min(coords[labels==k1,2])                         
                    
                    
                    # location X,Y
                    tmp5=np.min(coords[labels==k1,0])
                    tmp6=np.median(coords[labels==k1,0])
                    tmp7=np.max(coords[labels==k1,0])                                                                                          
                    
                    tmp8=np.min(coords[labels==k1,1])                    
                    tmp9=np.median(coords[labels==k1,1])
                    tmp10=np.max(coords[labels==k1,1])                                                                                          
                    
                    # two dimensional aspect ratio
                    tmp11=np.max(coords[labels==k1,3])                                             
                    
                else:
                    tmp1,tmp2,tmp3,tmp4=0,0,0,0
                    tmp5,tmp6,tmp7,tmp8=0,0,0,0
                    tmp9,tmp10,tmp11=0,0,0                    
        
                # add features to list    
                extracted_feature.append(tmp1)        
                extracted_feature.append(tmp2)        
                extracted_feature.append(tmp3)        
                extracted_feature.append(tmp4)        
                extracted_feature.append(tmp5)                
                extracted_feature.append(tmp6)                
                extracted_feature.append(tmp7)        
                extracted_feature.append(tmp8)                
                extracted_feature.append(tmp9)                
                extracted_feature.append(tmp10)                
                extracted_feature.append(tmp11)                
        else:
            extracted_feature=np.zeros(len(extracted_feature))
                
        #print extracted_feature
        extracted_features.append(extracted_feature)

    # stack all features
    x=np.stack(extracted_features)
    print 'collected features shape:', x.shape
    
    return x
    
    
#%%

path2dsb_nodes='/media/mra/My Passport/Kaggle/datascience2017/hpz440/LUNA2016/hdf5/nodes/nfolds_dsb_nodes.hdf5'
if os.path.exists(path2dsb_nodes):
    ff_node=h5py.File(path2dsb_nodes,'r')
    #ff_r2=h5py.File(path2dsb_features2,'r')
    print 'dsb nodes:', len(ff_node)
else:
    print 'does not exist'

path2dsbtest_nodes='/media/mra/My Passport/Kaggle/datascience2017/hpz440/LUNA2016/hdf5/nodes/nfolds_dsbtest_nodes.hdf5'
if os.path.exists(path2dsbtest_nodes):
    ff_node=h5py.File(path2dsbtest_nodes,'r')
    #ff_r2=h5py.File(path2dsb_features2,'r')
    print 'dsb nodes:', len(ff_node)
else:
    print 'does not exist'


# read dsb resample
path2dsb_resample=path2dsb_internal+'dsb_resampleXY.hdf5'
ff_dsb=h5py.File(path2dsb_resample,'r')
print 'dsb :', len(ff_dsb)
step=24

# load metadata
path2dsb_spacing=path2dsb_external+'dsb_spacing.hdf5'
ff_dsb_spacing=h5py.File(path2dsb_spacing,'r')
print 'total subjects:', len(ff_dsb_spacing)

# path 2 dsb test resample
path2dsbtest_resample=path2dsb_internal+'dsbtest_resampleXY.hdf5'
ff_dsbtest=h5py.File(path2dsbtest_resample,'r')
print 'total subjects:', len(ff_dsbtest)

# load metadata for dsb test
path2dsbtest_spacing=path2dsb_external+'dsbtest_spacing.hdf5'
ff_dsbtest_spacing=h5py.File(path2dsbtest_spacing,'r')
print 'total subjects:', len(ff_dsbtest_spacing)



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

# extract features
X_dsb=collect_features(df_train,'train')    
X_dsb_test=collect_features(df_test,'test')    
print X_dsb.shape
print X_dsb_test.shape

#%%

# train XGBOOST

y_dsb = df_train['cancer'].as_matrix()
n_folds=10

# train validation split
skf = list(StratifiedKFold(y_dsb, n_folds))

# xgboost classifier
clf = xgb.XGBRegressor(max_depth=4,
                           n_estimators=200,
                           min_child_weight=7,
                           learning_rate=0.07,
                           nthread=8,
                           subsample=0.70,
                           colsample_bytree=0.80,
                           seed=4241)

# train and validation
loss=[]
y_dsb_pred=np.zeros(len(X_dsb_test))
for i, (train, test) in enumerate(skf):
            print "Fold", i
            x_train = X_dsb[train]
            y_train = y_dsb[train]
            x_test = X_dsb[test]
            y_test = y_dsb[test]
            clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=True, eval_metric='logloss', early_stopping_rounds=20)
            y_pred=clf.predict(x_test,ntree_limit=clf.best_iteration)
            loss.append(log_loss(y_test,y_pred))
            y_dsb_pred+=clf.predict(X_dsb_test)

print 'average loss: %.3f' %(np.mean(loss))

# average
y_dsb_pred=y_dsb_pred/n_folds

# plot feature importance
plot_importance(clf)
plt.show()


#%%
df = pd.read_csv('../stage1_submission.csv')
print('Number of training patients: {}'.format(len(df)))
df.head()

df['cancer'] = y_dsb_pred
    
import datetime
now = datetime.datetime.now()
info='loss_'+str(np.mean(loss))
suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
sub_file = os.path.join('./output/submission', 'submission_' + suffix + '.csv')

df.to_csv(sub_file, index=False)
print(df.head())    


