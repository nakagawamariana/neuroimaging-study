# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 23:50:23 2020

@author: Maria
"""

import os
import numpy as np
import nibabel as nib #nos permite leer las imagenes
import matplotlib 
import matplotlib.pyplot as plt 
import skimage
from skimage import io
from skimage import filters
from nilearn import datasets
import modules
from nilearn import plotting

#Procesado y normalizacion de imagenes
'''----------------------Preprocesado+Normalizacion-------------------'''

def image_prep(img):
    
    #Vamos a probar los algoritmos con imágenes más pequeñas para disminuir el tiempo de computación
    a = np.array(skimage.transform.resize(img.dataobj, (100,140)))
    #a = np.array(skimage.transform.rescale(img.dataobj,0.3))   
    img_gray= a[:,:,128] #elegimos el corte
    
    #normalizo la imagen
    img_o=img_gray
    img_o=img_o/np.max(img_o)
    
    return img_o


#Función para optimizar código y comparar las imágenes de una manera más cómoda visualmente
'''----------------------Optimización Gauss-------------------'''
def comparar_gauss(img_gauss, iteraciones, threshold, cont):
    
    degauss_img = modules.aniso_filter(img_gauss, iteraciones, threshold) 
    
    fig = plt.figure(figsize=(10,10))
    
    plt.subplot(121)
    plt.imshow(img_gauss,cmap=plt.cm.gray)
    title1 = 'Gauss '+str(cont)
    plt.title(title1), plt.axis('off')
    
    plt.subplot(122)
    title2 = 'Aniso '+str(cont)
    plt.title(title2), plt.axis('off')
    plt.imshow(degauss_img, cmap=plt.cm.gray)


#Función para optimizar código y comparar las imágenes de una manera más cómoda visualmente
'''----------------------Optimización s&p-------------------'''
def comparar_salpimienta(img_salpimienta, iteraciones, threshold, cont):
    
    desalt_img = modules.aniso_filter(img_salpimienta, iteraciones, threshold) 
    
    fig = plt.figure(figsize=(10,10))
    
    plt.subplot(121)
    plt.imshow(img_salpimienta,cmap=plt.cm.gray)
    title1 = 'Sal y pimienta '+str(cont)
    plt.title(title1), plt.axis('off')
    
    plt.subplot(122)
    title2 = 'Aniso '+str(cont)
    plt.title(title2), plt.axis('off')
    plt.imshow(desalt_img, cmap=plt.cm.gray)
    
'''----------------------Batch algoritmos NLM---------------'''
    
def all_filters(img, nlm, sp, cpp1, cpp2, cpp3):
    img_pad = np.pad(img,1, mode='reflect')
    matriz_imagen1 = modules.nlm(img,img_pad, nlm)
    nlm_samepatch = modules.nlm_samepatch(img,img_pad, sp)
    nlm_cpp = modules.nlm_cpp(img, img_pad,cpp1, cpp2, cpp3)

    return matriz_imagen1, nlm_samepatch, nlm_cpp