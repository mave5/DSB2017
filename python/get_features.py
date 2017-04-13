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
        
    if (x<=w/2) & (y<=w/2):
        loc=0 #'top left'
    elif (x>w/2) & (y<=w/2):
        loc=1 #'top right'
    elif (x<=w/2) & (y>w/2):
        loc=2 #'bottom left'        
    elif (x>w/2) & (y>w/2):
        loc=3 #'bottom right'        
    return loc,(z*step,y*step,x*step)

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
start_point=700
end_point=1397
y_cancer=y[start_point:end_point]

cnz=[]
cz=[]
c1s=[]
c2s=[]
c3s=[]
c4s=[]
c5s=[]
fig = plt.figure()
cc=['r','b','g','k','y','c']
mm=['o','>','+','*','v','>']
lg=["4 to 8 mm","8 to 12 mm","12 to 16 mm","16 to 20 mm","20-24 mm"]
for k,id in enumerate(df_train.id[start_point:end_point]):
    print id,y_cancer[k]

    # read group features and labels    
    grp=ff_r[id]
    feature=grp['features']#.value
    # max pool over 2*2*2
    feature=np.max(feature,axis=(2,3,4))
    
    # get output
    y_pred=np.argmax(grp['output'],axis=1)
    #sort_i=np.argsort(y_pred)        
    nz_i=np.nonzero(y_pred)
    
    N,H,W=ff_dsb[id]['X'].shape
    
    # path to csv file to save scores
    #path2csv = './output/features/features'+id+'.csv'
    #first_row = 'id'
    #with open(path2csv, 'w+') as f:
        #f.write(',id: '+id+ '   cancer: '+ str(y_cancer[k])+'\n')
        #f.write(',,' + 'Z,Y,X'+'\n')
        #f.write(',image size:,' + str((N,H,W))+'\n')
    
    
    # probability
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
        features.append(feature[ind])
        
        reg,zyx=chunknb2zyx(ind,(N,H,W,step))
        #print reg,zyx,node_size
        regs.append(reg)
        coords.append(zyx)
        shapes=(N,H,W)
        #with open(path2csv, 'a') as f:
            #f.write('feature: '+str(ind)+'  diameter:'+str(4*y_pred[ind])+',location:,'+str(zyx)+'\n')        
            #f.write(str(feature[ind])+'\n')        

   
    # plot coords
    coords=np.array(coords)
    top_nodesize=np.array(top_nodesize)
    #fig = plt.figure()    
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(coords[:,2], coords[:,1], coords[:,0],c='r')
    
    for ns in range(1,5):
        xs=coords[top_nodesize==ns,2]
        ys=coords[top_nodesize==ns,1]
        zs=coords[top_nodesize==ns,0]
        ax.scatter(xs, ys, zs,c=cc[ns-1],label=ns,marker=mm[ns-1])
        

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1],loc=2)
    
    ax.axis([0,300,0,300])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('cancer: '+str(y_cancer[k]))
    plt.title('cancer: '+str(y_cancer[k]))
    fig.savefig('./output/features/figs/'+id+'.jpg')

    # PCA
    #pca = PCA(n_components=10)
    #pca.fit(feature)
    #print(pca.explained_variance_ratio_) 


    # clustering
    #kmeans_model = KMeans(n_clusters=2, random_state=1).fit(coords)
    #labels = kmeans_model.labels_
    #print np.count_nonzero(labels==0),np.count_nonzero(labels==1)#,np.count_nonzero(labels==2),np.count_nonzero(labels==3)

    # find relative distance
    #coords1=coords[labels==1]
    #coords0=coords[labels==0]

    #print np.sum(top_nodesize),np.sum(top_nodesize[labels==0]),np.sum(top_nodesize[labels==1])


dad

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


    
    