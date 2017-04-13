import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import cv2
import utils
#import scipy.ndimage as ndimage
#from sklearn.cross_validation import KFold
#from skimage import measure, morphology, segmentation
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import dicom
import h5py
#%matplotlib inline
p = sns.color_palette()
#import time

# get package versions
def get_version(*vars):
    for var in vars:
        module = __import__(var)    
        print '%s: %s' %(var,module.__version__)
    
# package version    
get_version('numpy','matplotlib','cv2','sklearn','skimage','scipy')
#%%

path2data='../sample_images/'
#path2data='/media/mra/My Passport/Kaggle/datascience2017/data/stage1/'
path2data="/home/mra/Desktop/DSB/stage2/"
patients=os.listdir(path2data)
patients.sort()
print len(patients)

# resize
h,w=512,512
#%%

# Load dicom files
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness

    RefDs=slices[0]
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))        
    return slices,ConstPixelSpacing
    
def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    #image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)    
#%%
print 'wait ...'
nb_dicoms=[]
for d in patients:
    #print("Patient '{}' has {} scans".format(d, len(os.listdir(path2data + d))))
    nb_dicoms.append(len(os.listdir(path2data + d)))

utils.array_stats(nb_dicoms)    
print('Total patients {} Total DCM files {}'.format(len(patients), len(glob.glob(path2data+'*/*.dcm'))))

plt.figure(figsize=((10,5)))
plt.hist(nb_dicoms, color=p[2])
plt.ylabel('Number of patients')
plt.xlabel('DICOM files')
plt.title('Histogram of DICOM count per patient')
plt.show()

#%%

path2csv= "/media/mra/win71/data/misc/kaggle/datascience2017/stage2_sample_submission.csv"
df_test = pd.read_csv(path2csv)
print('Number of training patients: {}'.format(len(df_test)))
print('Cancer rate: {:.4}%'.format(df_test.cancer.mean()*100))
df_test.head()

path2stage2="/media/mra/win71/data/misc/kaggle/datascience2017/data/stage2.hdf5"
ff_dsb_stage2=h5py.File(path2stage2,'w-')


for k,p_id in enumerate(df_test.id):
    scan,spacing=load_scan(path2data+p_id)
    X = get_pixels_hu(scan)
    print k,p_id
    print spacing,X.shape,X.dtype
    
    grp=ff_dsb_stage2.create_group(p_id)
    grp['X']=X
    grp['spacing']=spacing[::-1]
    

ff_dsb_stage2.close()
print 'data saved!'

# verify
ff_dsb_stage2=h5py.File(path2stage2,'r')
print 'total:', len(ff_dsb_stage2)
ff_dsb_stage2.close()


#%%

