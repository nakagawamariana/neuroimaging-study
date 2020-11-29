# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 11:35:21 2020
@author: nakag
"""

import scipy
import numpy as np
import nibabel as nib

import matplotlib

import matplotlib.pyplot as plt 
import skimage
from skimage import io
from skimage import measure
from skimage import filters
from nilearn import datasets
from skimage.color import rgb2gray
from skimage.transform import resize 
from imimposemin import imimposemin
from skimage.segmentation import watershed

    

def RegionGrowingP2(img, umbral_inf, umbral_sup):
    '''
    Parameters
    ----------
    img : Array of float32
        Imagen original
    umbral_inf : float
        Umbral por debajo del nivel de gris del punto seleccionado en seed.
    umbral_sup : float
        Umbral por encima del nivel de gris del punto seleccionado en seed.
    Returns
    -------
    region : Array of float64
        ROI de la imagen deseada.
    '''
    plt.figure()
    plt.imshow(img, cmap='gray') #Hacemos la representación la imagen para poder decidir donde posicionar la semilla
    click_markers = plt.ginput(n=1)  #Utilizamos la funcion .ginput() para posicionamiento de semilla
    plt.close()
    click_markers = list(click_markers[0]) #Transformamos a una lista

         
    markers = [round(num) for num in click_markers ] #Redondeamos a números enteros
    seed = [markers[1],markers[0]] #Cambiamos el orden de los elementos para poder utilizarlo como coordenadas
     
    
    print('Las coordenadas de las semillas son: ', seed)
    
    coords = np.array([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]) #Lista de coordenadas para hacer las comparaciones con los píxeles adyacentes
    
    #Pasar nuestra semilla a un array
    pixels = np.array(seed)
    
    #crear una matriz de ceros
    region= np.zeros(shape=(img.shape[0],img.shape[1]))
    
    #guardar nuestro pixel semilla en la ROI y lo llevamos a blanco
    region[pixels[0],pixels[1]]=1
    #definimos nuestro intervalo para decidir si incluirlo en la ROI o no

    intervalo_inf = img[pixels[0],pixels[1]]-umbral_inf
    intervalo_sup = img[pixels[0],pixels[1]]+umbral_sup
    
    #Parseamos por la lista coords para ir haciendo la conectividad a 8.
    for x in range (0, coords.shape[0]):
        #Dfinimos una condicion en la que si la intensidad del pixel comparado se encuentra entre el intervalo de similitud
        if intervalo_inf<=img[pixels[0]+coords[x,0], pixels[1]+coords[x,1]]<=intervalo_sup :
                                
            region[pixels[0]+coords[x,0], pixels[1]+coords[x,1]] = 1     #Se añade a la ROI sustituyendo en dichas coordenadas por un 1  (se lleva a blanco)
            
        else:
            pass #Si no cumple la condición se prosigue con la comparación
                    
    new_pix = np.where(region == 1) #Buscamos aquellas coordenadas en las que la matriz region es igual a 1
    Coordinates = np.array(list(zip(new_pix[0], new_pix[1]))) #Creamos un array con dichas coordenadas
    listOfCoordinates = list(zip(new_pix[0], new_pix[1])) #lista de las coordenadas
    regionCoords =[] #creamos una lista vacía para realizar la comparación
    
                
    while len (listOfCoordinates)!=len(regionCoords): #COmparamos las listas de coordenadas de los puntos en los que la región es 1, antes y después de realizar cada iteración de conectividad a 8
        new_pix = np.where(region == 1)#Buscamos aquellas coordenadas en las que la matriz region es igual a 1
        Coordinates = np.array(list(zip(new_pix[0], new_pix[1])))  #Creamos un array con dichas coordenadas
        listOfCoordinates = list(zip(new_pix[0], new_pix[1])) #lista de las coordenadas
        
        for i in range (0, Coordinates.shape[0]): #Para todas las coordenadas de los píxeles de la región creamos un bucle for
        
                for x in range (0, 8):
                    if Coordinates[i,0]+coords[x,0] >= 0 and Coordinates[i,1]+coords[x,1]>= 0 and Coordinates[i,0]+coords[x,0]<img.shape[0] and Coordinates[i,1]+coords[x,1]<img.shape[1]: #Creamos una condición para evitar que el algoritmo de error al comparar píxeles de los bordes de la imagen. 
                    #Esto lo conseguimos imponiendo que las comparaciones no se hagan sobre coordenadas negativas ni fuera de rango
                        if intervalo_inf<=img[Coordinates[i,0]+coords[x,0], Coordinates[i,1]+coords[x,1]]<=intervalo_sup :
                                        
                            region[Coordinates[i,0]+coords[x,0], Coordinates[i,1]+coords[x,1]] = 1      
                            #Volvemos a hacer la misma comparación que en la primera iteración explicada fuera del bucle while
                        else:
                            pass
                    else:
                        pass
        regionCoords = np.where(region == 1)#Volvemos a evaluar la nueva roi con cada iteración.
        regionCoords = list(zip(regionCoords[0], regionCoords[1]))#Convertimos a lista

    #Devolvemos la ROI final como resultado de la función
    return region            

         

def WatershedExerciseP2(img, numberofseeds):
    '''
    
    Parameters
    ----------
    img : Array of float32
        Imagen original.
    numberofseeds : int
        Número de semillas que vamos a utilizar.
    Returns
    -------
    watershed1 : Array of int32
        Máscara de regiones tras algoritmo Watershed con img_sobel como input.
    watershed2 : Array of int32
        Máscara de regiones tras algoritmo Watershed con minimos como input..
    '''
    white_dots= np.zeros(shape=(img.shape[0],img.shape[1])) #Matriz de ceros a la que vamos a añadir los mínimos locales introducidos con .ginput()
    img_sobel=filters.sobel(img)  #Transformamos la imagen original a gradiente mediante el algoritmo de Sobel

    plt.figure()
    plt.title('Semillas')
    plt.imshow(img, cmap='gray') #Hacemos la representación la imagen para poder decidir donde posicionar las semillas
    click_markers = plt.ginput(n=numberofseeds) #Utilizamos la funcion .ginput() para posicionamiento de samillas, nótese que el número de estas lo introducimos como parámetro de la función
    plt.close()
    clicks = [(sub[1], sub[0]) for sub in click_markers] #cambiamos el orden de las tuplas obtenidas en .ginput() para poder usarlas como coordenadas.
    markers = np.array(clicks,dtype = int) #transformamos la lista de tuplas a un array y pasamos de elementos float a int
    
    print('Las coordenadas de las semillas son: ', markers)

    
    white_dots[markers[:,0], markers[:,1]] = 1 #Sustituímos los puntos que hemos almacenado en la variable markers por 1s en la matriz de ceros creada al inicio de nuestra función
   
    
    minimos = imimposemin(img_sobel, white_dots) #modifica la imagen de la máscara en escala de grises utilizando la reconstrucción morfológica por lo que sólo tiene mínimo regional donde la imagen de marcador binario es distinto de cero
    
    watershed1= watershed(img_sobel) #Aplicamos el algoritmo de Watershed sobre imagen de Sobel 
    watershed2 = watershed(minimos) #Aplicamos el algoritmo de Watershed con los mínimos regionales
    
    return watershed1, watershed2


def anisodiff(img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),option=1,plot_flag=False):
        """
        Anisotropic diffusion.
 
        Usage:
        imgout = anisodiff(im, niter, kappa, gamma, option)
 
        Arguments:
                img    - input image
                niter  - number of iterations
                kappa  - conduction coefficient 20-100 ?
                gamma  - max value of .25 for stability
                step   - tuple, the distance between adjacent pixels in (y,x)
                option - 1 Perona Malik diffusion equation No 1
                         2 Perona Malik diffusion equation No 2
                plot_flag - if True, the image will be plotted
 
        Returns:
                imgout   - diffused image.
 
        kappa controls conduction as a function of gradient.  If kappa is low
        small intensity gradients are able to block conduction and hence diffusion
        across step edges.  A large value reduces the influence of intensity
        gradients on conduction.
 
        gamma controls speed of diffusion (you usually want it at a maximum of
        0.25)
 
        step is used to scale the gradients in case the spacing between adjacent
        pixels differs in the x and y axes
 
        Diffusion equation 1 favours high contrast edges over low contrast ones.
        Diffusion equation 2 favours wide regions over smaller ones.
 
        Reference:
        P. Perona and J. Malik.
        Scale-space and edge detection using ansotropic diffusion.
        IEEE Transactions on Pattern Analysis and Machine Intelligence,
        12(7):629-639, July 1990.
 
        Original MATLAB code by Peter Kovesi  
        School of Computer Science & Software Engineering
        The University of Western Australia
        pk @ csse uwa edu au
        <http://www.csse.uwa.edu.au>
 
        Translated to Python and optimised by Alistair Muldal
        Department of Pharmacology
        University of Oxford
        <alistair.muldal@pharm.ox.ac.uk>
 
        June 2000  original version.      
        March 2002 corrected diffusion eqn No 2.
        July 2012 translated to Python
        """
 
        # initialize output array
        img = img.astype('float64')
        imgout = img.copy()
 
        # initialize some internal variables
        deltaS = np.zeros_like(imgout)
        deltaE = deltaS.copy()
        NS = deltaS.copy()
        EW = deltaS.copy()
        gS = np.ones_like(imgout)
        gE = gS.copy()    
 
        for ii in range(niter):
 
                # calculate the diffs
                deltaS[:-1,: ] = np.diff(imgout,axis=0)
                deltaE[: ,:-1] = np.diff(imgout,axis=1)
 
                # conduction gradients (only need to compute one per dim!)
                if option == 1:
                        gS = np.exp(-(deltaS/kappa)**2.)/step[0]
                        gE = np.exp(-(deltaE/kappa)**2.)/step[1]
                elif option == 2:
                        gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
                        gE = 1./(1.+(deltaE/kappa)**2.)/step[1]
 
                # update matrices
                E = gE*deltaE
                S = gS*deltaS
 
                # subtract a copy that has been shifted 'North/West' by one
                # pixel. don't as questions. just do it. trust me.
                NS[:] = S
                EW[:] = E
                NS[1:,:] -= S[:-1,:]
                EW[:,1:] -= E[:,:-1]
 
                # update the image
                imgout += gamma*(NS+EW)
 
                               
        if plot_flag:
             # create the plot figure, if requested
            plt.figure(figsize=(12, 5))
            plt.subplot(121)
            plt.imshow(img, cmap=plt.cm.gray)
            plt.title('Original image'), plt.axis('off')
            plt.subplot(122)
            plt.imshow(imgout, cmap=plt.cm.gray)
            plt.title('Filtered image (Anisotropic Diffusion)'), plt.axis('off')
 
        return imgout

def WatershedNoGinput(img, markers):
    '''
    
    Parameters
    ----------
    img : Array of float32
        Imagen original.
    markers : array
        Array de semillas.
    Returns
    -------
    watershed1 : Array of int32
        Máscara de regiones tras algoritmo Watershed con img_sobel como input.
    watershed2 : Array of int32
        Máscara de regiones tras algoritmo Watershed con minimos como input..
    '''
    white_dots= np.zeros(shape=(img.shape[0],img.shape[1])) #Matriz de ceros a la que vamos a añadir los mínimos locales introducidos con .ginput()
    img_sobel=filters.sobel(img)  #Transformamos la imagen original a gradiente mediante el algoritmo de Sobel


    
    white_dots[markers[:,0], markers[:,1]] = 1 #Sustituímos los puntos que hemos almacenado en la variable markers por 1s en la matriz de ceros creada al inicio de nuestra función
 
    
    minimos = imimposemin(img_sobel, white_dots) #modifica la imagen de la máscara en escala de grises utilizando la reconstrucción morfológica por lo que sólo tiene mínimo regional donde la imagen de marcador binario es distinto de cero
    
    watershed1= watershed(img_sobel) #Aplicamos el algoritmo de Watershed sobre imagen de Sobel 
    watershed2 = watershed(minimos) #Aplicamos el algoritmo de Watershed con los mínimos regionales
    
    return watershed1, watershed2





    
