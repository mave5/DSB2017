import numpy as np
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from glob import glob
from sklearn.externals import joblib
import cv2
import matplotlib.pylab as plt
from sklearn import cross_validation
#%%
working_path = "./output/numpy/luna/allsubsets/"

file_list=glob(working_path+"images_*.npy")
print 'total images %s' %(len(file_list))

mask_list=glob(working_path+"masks_*.npy")
print 'total images %s' %(len(mask_list))

lungmask_list=glob(working_path+"lungmask_*.npy")
print 'total images %s' %(len(lungmask_list))
#%%

def grays_to_RGB(img):
    # turn 2D grayscale image into grayscale RGB
    return np.dstack((img, img, img))


def image_with_mask(img, mask,color=(1,0,0)):
    # bring to [0,1]
    img=(img-np.min(img)*1.)/np.max(img)    
    
    mask=np.asarray(255*mask,dtype='uint8') 

    # returns a copy of the image with edges of the mask added in red
    if len(img.shape)==2:	
	img_color = grays_to_RGB(img)
    else:
	img_color =img

    mask_edges = cv2.Canny(mask, 100, 200) > 0
    img_color[mask_edges, 0] = color[0]  # set channel 0 to bright red, green & blue channels to 0
    img_color[mask_edges, 1] = color[1]
    img_color[mask_edges, 2] = color[2]
    return img_color

def disp_img_2masks(img,mask1,mask2,r=1,c=1,d=0,indices=None):
    if mask1 is None:
        mask1=np.zeros(img.shape,dtype='uint8')
    if mask2 is None:
        mask2=np.zeros(img.shape,dtype='uint8')
        
    N=r*c    
    if d==2:
        # convert to N*C*H*W
        img=np.transpose(img,(2,0,1))
        img=np.expand_dims(img,axis=1)
        
        mask1=np.transpose(mask1,(2,0,1))
        mask1=np.expand_dims(mask1,axis=1)

        mask2=np.transpose(mask2,(2,0,1))
        mask2=np.expand_dims(mask2,axis=1)
        
    if indices is None:    
        # random indices   
        n1=np.random.randint(img.shape[0],size=N)
    else:
        n1=indices
    
    I1=img[n1,0]
    #M1=mask1[n1,0]
    M1=np.zeros(I1.shape,dtype='uint8')
    for c1 in range(mask1.shape[1]):
        M1=np.logical_or(M1,mask1[n1,c1,:])    
    #M2=mask2[n1,0]
    M2=np.zeros(I1.shape,dtype='uint8')
    for c1 in range(mask2.shape[1]):
        M2=np.logical_or(M2,mask2[n1,c1,:])    
    
    C1=(1,0,0)
    C2=(0,0,1)
    for k in range(N):    
        imgmask=image_with_mask(I1[k],M1[k],C1)
        imgmask=image_with_mask(imgmask,M2[k],C2)
        plt.subplot(r,c,k+1)
        plt.imshow(imgmask)
        plt.title(n1[k])
    plt.show()            
    return n1        

# train data collection

def stack_data(file_list):
    X=[]
    Y=[]
    for sbnb,fname in enumerate(file_list):
        print "working on file: %s, %s" %(sbnb,fname)
    
        X0 = np.load(fname) # load image
        Y0 = np.load(fname.replace("images","masks")) # load nodule mask
        X.append(X0) # N*H*W
        Y.append(Y0) # N*H*W

    # stack array
    X=np.vstack(X).astype('int16')
    Y=np.vstack(Y).astype('uint8')
    
    # convert to N*1*H*W
    X=X[:,np.newaxis,:]
    Y=Y[:,np.newaxis,:]
    
    return X,Y


#%%

# split and stack data to train and test 
file_list=glob(working_path+"images_*.npy")
print 'total lung masks %s' %(len(lungmask_list))


trn_fl, val_fl, _, _ = cross_validation.train_test_split(file_list, file_list, random_state=420, 
                                                                   test_size=0.1)
print 'number of train masks %s' %(len(trn_fl))
print 'number of val masks %s' %(len(val_fl))

#%% stack train data
X_train,Y_train=stack_data(trn_fl)    
joblib.dump(X_train,working_path+"trainX.joblib")
joblib.dump(Y_train,working_path+"trainY.joblib")

# stack test data
X_test,Y_test=stack_data(val_fl)    
joblib.dump(X_test,working_path+"testX.joblib")
joblib.dump(Y_test,working_path+"testY.joblib")   

#%% verify

prefix='test'
X=joblib.load(working_path+prefix+"X.joblib")
print X.shape

Y=joblib.load(working_path+prefix+"Y.joblib")
print Y.shape

plt.figure(figsize=(20,100))
n1=disp_img_2masks(X,Y,None,3,4,True)
#%%
