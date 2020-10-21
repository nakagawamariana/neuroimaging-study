# -*- coding: utf-8 -*-
"""

@author: nakagawa mariana
"""


import os
import numpy as np
import nibabel as nib
import matplotlib 
import matplotlib.pyplot as plt 
from skimage import io
from skimage import filters
from nilearn import datasets

img = nib.load(r'\Users\Escritorio\squizo\sub-01\anat\sub-01_T1w.nii.gz')

print (img)

affine = img.affine
print(affine) 

header = img.header['pixdim']
print(header)

print( img.get_data_dtype())

#plt.imshow(img)
from nilearn import plotting
plotting.plot_img(img, title="Prueba1")

plotting.show()

def add_noise(n_type,image,porcentaje):
    if n_type=='gauss':
        gaussian_noise=np.random.normal(loc=0.0, scale=1.0, size=np.shape(image))
        noisy = image + gaussian_noise
        return noisy
    elif n_type=='s&p' :
        cant = 0.004
        ruido_output= np.copy(image)

        #ruido de sal
        num_salt = np.ceil(cant * image.size * 0.5)
        pos = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        ruido_output[pos]=1
        #ruido pimienta
        num_pepper= np.ceil(cant * image.size * 0.5)
        pos = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        ruido_output[pos]=0
        return ruido_output
    

img_data = img.get_data()
print (img_data)
logical_mask =img_data == 1  # force the mask to be logical type
mean = img_data[logical_mask].mean()
std = img_data[logical_mask].std()
normalized = nib.Nifti1Image((img_data - mean) / std, img.affine, img.header)


img_gauss=add_noise('gauss',normalized,0.5)
img_salt_pepper=add_noise('s&p',normalized,0.5)