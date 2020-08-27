#Imports
import numpy as np
from skimage import io
import os
import scipy
import scipy.ndimage

#Image specifications
image_size = 200
image_size_s = int(image_size/10)

#Path specifications
#origin_path = os.getcwd()
origin_path = '/Users/Simona' #path to tif file
origin_path = '/Users/Simona/Notebooks_truebranch'
os.chdir(origin_path)

sentinel  = io.imread('Sentinel_image_27_08.tif')
naip = io.imread('NAIP_image_27_08.tif')
#naip = sentinel

print(sentinel.shape)
print(naip.shape)

sentinel_crop = []
naip_crop = []

def tif_images(naip,sentinel,image_size,image_size_s):
    for i in range(0,int(naip.shape[0]/image_size)):
        naip_crop = naip[i*image_size:image_size+i*image_size,i*image_size:image_size+i*image_size,:]
        sentinel_crop = sentinel[i*image_size_s:image_size_s+i*image_size_s,i*image_size_s:image_size_s+i*image_size_s,:]
        sentinel_resized = scipy.ndimage.zoom(sentinel_crop,(10,10,1), order=0)
        temp_path = str(i)
        os.makedirs(origin_path+"/"+temp_path)
        os.chdir(origin_path+"/"+temp_path)
        io.imsave('sentinel_'+str(i)+'.png', sentinel_resized)
        io.imsave('naip_'+str(i)+'.png', naip_crop)

tif_images(naip,sentinel,image_size,image_size_s)
