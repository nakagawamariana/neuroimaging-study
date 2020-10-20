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