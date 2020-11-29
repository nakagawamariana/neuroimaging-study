# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 22:25:34 2020
@author: nakag
"""
#
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt 
import scipy.ndimage.filters as filters
from numba import njit, prange

#Adición de ruido a las imágenes originales
'''----------------------Ruido Gaussiano-------------------'''
def add_gnoise(image,sigma):
    '''
    Parameters
    ----------
    image : Array of float32
        Imagen a la que queremos añadir ruido.
    sigma : float
        Parámetro que modula la cantidad de ruido gausiano que añadimos (desviación típica de la distribución normal).
    Returns
    -------
    noisy : Array of float64
        Imagen con ruido gaussiano.
    '''

    gaussian_noise=np.random.normal(loc=0.0, scale=sigma,size=np.shape(image))#Creación del ruido mediante una distribución normal a la que entra sigma como parámetro.
    noisy = image + gaussian_noise#Adición de ruido gaussiano a la imagen original
    return noisy

'''----------------------Ruido Impulsivo-------------------'''
def salpimienta(image,intensity):
    '''
    Parameters
    ----------
    image : Array of float32
        Imagen a la que queremos añadir ruido.
    intensity : float
        Parámetro que modula la cantidad de ruido impulsivo que añadimos.
    Returns
    -------
    ruido_output: Array of float64
        Imagen con ruido gaussiano.
    '''
    cant = intensity
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
    


def mean_filter(img,size_filter):

    # the filter is divided by size_filter^2 for normalization
    mean_filter = np.ones((size_filter,size_filter))/np.power(size_filter,2)
    # performing convolution
    img_meanfiltered = filters.convolve(img, mean_filter,mode='reflect')
    return img_meanfiltered


def median_filter(img, size):
    img_filtered = filters.median_filter(img,size = size,mode='reflect')
    return img_filtered

def gaussian_filter(img,sigma):
    # performing convolution
    img_gaussianfiltered = filters.gaussian_filter(img, sigma=sigma,mode='reflect')
    return img_gaussianfiltered


'''----------------------Filtro NLM-------------------'''    
#NLM: filtrar la imagen mediante el promedio ponderado de 
#los diferentes píxeles de la imagen en función de su similitud
@njit(parallel=True)
def nlm(img_ori,img_pad, h_square): 
    '''
    Parameters
    ----------
    img_ori : Array of float64
        Imagen ruidosa a filtrar.
    img_pad: Array of float64
        Imagen ruidosa con padding.
    h_square : float
        Parámetro de similitud asociado al grado de filtrado.
        
    Returns
    -------
    matriz_imagen : Array of float64
        Imagen filtrada mediante NLM.
    '''
    matriz_imagen = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1])) #Creamos otra matriz de ceros en la que indexaremos la imagen final

    for i in prange(1,img_pad.shape[0]-1):#Vamos a crear dos parches 3x3 iterando sobre la imagen que vamos a ir comparando 
        for j in prange(1,img_pad.shape[1]-1):
            
            #parche 3x3 de referencia (el que se quiere comparar con el resto de parches de la imagen, estará centrado en el pixel a filtrar)
            #Nótese que nuestro pixel central está en la posición [i,j]
            matriz1 = np.array([[img_pad[i-1,j-1],img_pad[i-1,j],img_pad[i-1,j+1]], 
                                [img_pad[i,j-1],img_pad[i,j],img_pad[i,j+1]], 
                                [img_pad[i+1,j-1],img_pad[i+1,j],img_pad[i+1,j+1]]])
            
            
            # Aquí se inicializa la matriz de pesos para que no surjan errores al implementar numba
            matriz_pesos = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1])) #almacenamos los pesos en una matriz
            for x in prange(1,img_pad.shape[0]-1):
                for y in prange(1,img_pad.shape[1]-1):
                    
                    #parche 3x3 que va recorriendo la imagen para compararse con el primero
                    
                        matriz2 = np.array([[img_pad[x-1,y-1],img_pad[x-1,y],img_pad[x-1,y+1]], 
                                [img_pad[x,y-1],img_pad[x,j],img_pad[x,y+1]], 
                                [img_pad[x+1,y-1],img_pad[x+1,y],img_pad[x+1,y+1]]])
                    
                    #Cálculo de la distancia euclídea
                    
                        distance = np.sqrt((matriz1-matriz2)**2)
                        distance = np.sum(distance)
                    #Ponderación de cada píxel en función de su similitud respecto al pixel a filtrar 
                        weights_ij = (np.exp(-distance/h_square)) #Nótese que el parámetro h_square va asociado al grado de filtrado
                    
                        matriz_pesos[x-1,y-1] = weights_ij #Introducimos cada uno de los peso a la matriz
                    
            
            #Ponderación de máscara obtenida (Aplicamos en este paso la cte de normalización Z(i))
            matriz_ponderada = matriz_pesos/np.sum(matriz_pesos)
            
            #Finalmente se aplica la máscara a la imagen original
            matriz_imagen[i-1,j-1] = np.sum(np.multiply(img_ori,matriz_ponderada))
       
    return matriz_imagen

'''----------------------Filtro NLM modificación 1-------------------'''   
#Función de NLM ponderando el parche original
@njit(parallel=True)
def nlm_samepatch(img_ori, img_pad, h_square):
    '''
    
    Parameters
    ----------
    img_ori : Array of float64
        Imagen ruidosa a filtrar.
    img_pad: Array of float64
        Imagen ruidosa con padding.
    h_square : float
        Parámetro de similitud asociado al grado de filtrado.

    Returns
    -------
    matriz_imagen : Array of float64
        Imagen filtrada mediante NLM (comparación de píxeles centrales).

    '''
    matriz_imagen = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1]))#Creamos otra matriz de ceros en la que indexaremos la imagen final

    for i in prange(1,img_pad.shape[0]-1):#Vamos a crear dos parches 3x3 iterando sobre la imagen que vamos a ir comparando

        for j in prange(1,img_pad.shape[1]-1):
            
              #parche 3x3 de referencia (el que se quiere comparar con el resto de parches de la imagen, estará centrado en el pixel a filtrar)
              #Nótese que nuestro pixel central está en la posición [i,j]
            matriz1 = np.array([[img_pad[i-1,j-1],img_pad[i-1,j],img_pad[i-1,j+1]], 
                                [img_pad[i,j-1],img_pad[i,j],img_pad[i,j+1]], 
                                [img_pad[i+1,j-1],img_pad[i+1,j],img_pad[i+1,j+1]]])
            
            # Aquí se inicializa la matriz de pesos para que no surjan errores al implementar numba
            matriz_pesos = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1])) 
            for x in prange(1,img_pad.shape[0]-1):
                for y in prange(1,img_pad.shape[1]-1):
                    
                    #parche 3x3 que va recorriendo la imagen para compararse con el primero
                        matriz2 = np.array([[img_pad[x-1,y-1],img_pad[x-1,y],img_pad[x-1,y+1]], 
                                [img_pad[x,y-1],img_pad[x,y],img_pad[x,y+1]], 
                                [img_pad[x+1,y-1],img_pad[x+1,y],img_pad[x+1,y+1]]])
                    
                    #Cálculo de la distancia euclídea
                        distance = np.sqrt((matriz1-matriz2)**2)
                        distance = np.sum(distance)
                    #Ponderación de cada píxel en función de su similitud respecto al pixel a filtrar  
                        weights_ij = (np.exp(-distance/h_square))#Nótese que el parámetro h_square va asociado al grado de filtrado

                        matriz_pesos[x-1,y-1] = weights_ij #Introducimos cada uno de los peso a la matriz
                    
            matriz_pesos[i-1,j-1] = 0 #hago que nuestro pixel sea el de menor valor, para evitar que asuma que el máximo es la propia comparación consigo mismo
                        
            matriz_pesos[i-1,j-1] = np.max(matriz_pesos) #obtenemos el valor máximo de similitud que se haya encontrado en el resto de la imagen
            
            #Ponderación de máscara obtenida (Aplicamos en este paso la cte de normalización Z(i))
            matriz_ponderada = matriz_pesos/np.sum(matriz_pesos)
            
             #Finalmente se aplica la máscara a la imagen original
            matriz_imagen[i-1,j-1] = np.sum(np.multiply(img_ori,matriz_ponderada))
       
    return matriz_imagen


'''----------------------Filtro NLM-cpp modificación 2-------------------'''  
@njit(parallel=True)
def nlm_cpp(img_ori, img_pad, h_square, D_0, alpha):
    '''

    Parameters
    ----------
    img_ori : Array of float64
        Imagen ruidosa a filtrar.
    img_pad: Array of float64
        Imagen ruidosa con padding.
    h_square : float
        Parámetro de similitud asociado al grado de filtrado.
    D_0 : float
        Parámetro de la función eta que calcula la similitud entre píxeles
    alpha : float
        Parámetro de la función eta que calcula la similitud entre píxeles

    Returns
    -------
    matriz_imagen : Array of float64
        Imagen filtrada mediante NLM (comparación de píxeles centrales).
    '''

    matriz_imagen = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1]))#Creamos otra matriz de ceros en la que indexaremos la imagen final
    
    for i in prange(1,img_pad.shape[0]-1):#Vamos a crear dos parches 3x3 iterando sobre la imagen que vamos a ir comparando
        for j in prange(1,img_pad.shape[1]-1):
            
             #parche 3x3 de referencia (el que se quiere comparar con el resto de parches de la imagen, estará centrado en el pixel a filtrar)
              #Nótese que nuestro pixel central está en la posición [i,j]            
            matriz1 = np.array([[img_pad[i-1,j-1],img_pad[i-1,j],img_pad[i-1,j+1]], 
                                [img_pad[i,j-1],img_pad[i,j],img_pad[i,j+1]], 
                                [img_pad[i+1,j-1],img_pad[i+1,j],img_pad[i+1,j+1]]])
            
            '''
            Al igual que pasa con matriz_pesos, matriz_nu se debe inicializar
            antes de los bucles x,y, no antes de i,j.
            '''
            # Aquí se inicializa la matriz de pesos y la matriz nu para que no surjan errores al implementar numba
            matriz_pesos = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1])) #aqui almacenamos los pesos
            matriz_nu = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1])) #esta será la matriz que multiplicaremos por los pesos normalizados

            for x in prange(1,img_pad.shape[0]-1):
                for y in prange(1,img_pad.shape[1]-1):
                    
                    #parche 3x3 que va recorriendo la imagen para compararse con el primero
                        matriz2 = np.array([[img_pad[x-1,y-1],img_pad[x-1,y],img_pad[x-1,y+1]], 
                                [img_pad[x,y-1],img_pad[x,y],img_pad[x,y+1]], 
                                [img_pad[x+1,y-1],img_pad[x+1,y],img_pad[x+1,y+1]]])
                    
                    #Cálculo de la distancia euclídea
                        distance = np.sqrt((matriz1-matriz2)**2)
                        distance = np.sum(distance)
                    #Ponderación de cada píxel en función de su similitud respecto al pixel a filtrar                    
                        weights_ij = (np.exp(-distance/h_square))#Nótese que el parámetro h_square va asociado al grado de filtrado
                    #Ponderación de similitud entre píxeles centrales   
                        matriz_nu[x-1,y-1] = 1/(1+(np.abs(img_pad[i,j]-img_pad[x,y])/D_0)**(2*alpha))
                
                        matriz_pesos[x-1,y-1] = weights_ij#Introducimos cada uno de los peso a la matriz
                           
            #Ponderación de máscara obtenida (Aplicamos en este paso la cte de normalización Z(i))
            matriz_ponderada1 = matriz_pesos/np.sum(matriz_pesos)#normalización de los pesos
            
            matriz_nu_pond = np.multiply(matriz_nu,matriz_ponderada1)#ponderamos los pesos por nu, para que dependan de la similitud entre píxeles centrales
            
            matriz_ponderada_nu2 = matriz_nu_pond/np.sum(matriz_nu_pond)#normalización de los pesos tras ponderar por nu
            #Finalmente se aplica la máscara a la imagen original
            matriz_imagen[i-1,j-1] = np.sum(np.multiply(img_ori,matriz_ponderada_nu2))
       
    return matriz_imagen
 

'''----------------------Filtro Anisotrópico-------------------'''  

def aniso_filter(img, iteraciones, threshold):   
    
    '''
    Parameters
    ----------
    img : Array of float64
        Imagen ruidosa a filtrar.
        
    iteraciones : int
        Número de veces en las que se va a aplicar el filtrado
    threshold: float
        Umbral de gradiente que determina si se aplica o no el suavizado. 
        Está entre 0 y 1 porque la imagen está normalizada
        
    Returns
    -------
    values : Array of float64
        Imagen resultante del filtrado anisotrópico.
    '''
    
    #creamos un contador para contabilizar cuando hemos terminado la primera iteración
    cont=0
    
    #Inicializamos un bucle while teniendo en cuenta que se va a ejecutar siempre y cuando  
    #la variable cont sea menor que el de las iteraciones que hemos determinado
    
    while cont<iteraciones:
        if cont==0:  #este if solo se ejecuta durante la primera iteración
            
            #aplicamos sobel a una imagen con ruido
            img_sobel_g=filters.sobel(img)

            #realizamos el padding con img sobel
            img_sobel_g_pad=np.pad(img_sobel_g, 1, mode='reflect')
            
            #padding de la imagen original, es decir, la imagen con ruido
            img_noisy_pad=np.pad(img, 1, mode='reflect') 

            #matriz para almacenar 
            values=np.zeros(shape=(img_sobel_g.shape[0],img_sobel_g.shape[1]))
              
            cont=cont+1 #contabilizamos la primera iteración
            
        else: # se ejecuta cuando se supera la primera iteración, a partir de este momento se empiezan a aplicar los siguientes cambios 
        #sobre la matriz values (ya que queremos un resultado más pronunciado con cada iteración)
            
            #el procedimiento e
            
            #aplicamos sobel a una imagen con ruido
            img_sobel_g=filters.sobel(values)

            #realizamos el padding con img sobel
            img_sobel_g_pad=np.pad(img_sobel_g, 1, mode='reflect')

            #padding de la original, la imagen con ruido
            img_noisy_pad=np.pad(values, 1, mode='reflect') 
            
            #matriz para almacenar 
            values=np.zeros(shape=(img_sobel_g.shape[0],img_sobel_g.shape[1]))
            
            cont=cont+1  #sumamos una iteración más contabilizando que se ejecuta el proceso


        #aplicamos algoritmo anisotrópico
        
        #primero realizamos  dos bucles for para poder iterar por la imagen
        
        for i in range(1,img_sobel_g_pad.shape[0]-1):
            for j in range(1,img_sobel_g_pad.shape[1]-1):
                
                #creamos un parche 3x3 con el que guardar las intensidades de grises de la imagen 
                #Este parche irá iterando por toda la imagen
                
                parche_sobel = np.array([[img_sobel_g_pad[i-1,j-1],img_sobel_g_pad[i-1,j],img_sobel_g_pad[i-1,j+1]], 
                                       [img_sobel_g_pad[i,j-1],img_sobel_g_pad[i,j],img_sobel_g_pad[i,j+1]], 
                                        [img_sobel_g_pad[i+1,j-1],img_sobel_g_pad[i+1,j],img_sobel_g_pad[i+1,j+1]]])
                
                
                # definimos la variable gradiente como el sumatorio de las intensidades de la imagen de sobel.
                gradiente= np.sum(parche_sobel)
                    
                # ahora creamos un if para diferenciar las zonas por las que el algoritmo va a poder suavizar (zonas homogéneas)
                # y las zonas que no debe suavizar (bordes)
                
                if gradiente<threshold:  #el threshold es un float entre 0 y 1 porque la imagen está normalizada
                    
                    # si el gradiente es menor que el umbral definido entonces estamos ante una zona homogénea
                    # como es una zona homogénea y por tanto vamos a poder 'suavizar' la zona
                    
                    # creamos un parche 3x3 donde recoger las intensidades de gris de la imagen
                    parche_noisy = np.array([[img_noisy_pad[i-1,j-1],img_noisy_pad[i-1,j],img_noisy_pad[i-1,j+1]], 
                                       [img_noisy_pad[i,j-1],img_noisy_pad[i,j],img_noisy_pad[i,j+1]], 
                                        [img_noisy_pad[i+1,j-1],img_noisy_pad[i+1,j],img_noisy_pad[i+1,j+1]]])
                    
                    # calculamos la media de las intensidades del parche                
                    mean=np.mean(parche_noisy)
                    
                    # introducimos el valor de la intensidad media en la matriz de almacenamiento values, 
                    # en la coordenada [i-1, j-1]                   
                    values[i-1, j-1]=mean
                    
                else:
                    
                    # si el gradiente es mayor que el umbral entonces estamos ante un borde
                    # como es una zona de borde, no se 'suaviza', por lo que almacenamos en la matriz values
                    # los valores originales de la imagen con ruido en las coordenadas [i-1, j-1]
                    values[i-1, j-1]=img[i-1, j-1]
        
       

    #devolvemos la matriz values finalmente con la imagen resultante del filtrado anisotrópico
    return values  


#Función del filtro anisotrópico del script facilitado en aula virtual 

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

