# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 20:39:39 2022

@author: chaof
"""

# import packages
import numpy as np
from numpy.fft import fft2,fftshift,fftfreq
import PIL

# functions to be used
def Fringe(I0,V,f,x,y,phi):
    '''
    Inputs:
        I0: beam intensity
        V: visibility
        f: frequency
        x, y: position arguments
        phi: phase difference between 2 beams
    Output:
        Interference Pattern of 2 beams with the given arguments.
        Returns a matrix if guven (x,y) is matrix.
    '''
    return I0*(1+V*np.cos(2*np.pi*f*x*np.cos(phi)+2*np.pi*f*y*np.sin(phi)))

def Circular_Mask(d):
    '''
    Input:
        d: diametre of the aperture. 
            This should be the number of pixels of the given image.
    Output:
        arr: The circular mask. Returns 0  outside the aperture and 1 inside.
    '''
    r = d/2
    arr = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            if (i-r)**2+(j-r)**2 < r**2:
                arr[i,j] = 1
    return arr

def Zero_f_mask(n,epsilon=0.05):
    '''
    Input:
        epsilon: the amount of image to be covered by the mask. default = 0.05
        n: the diametre of the image = pixels of the image.
    Output:
        arr: the circular mask. 0 inside the circle and 1 outside.
    '''
    r = n/2
    arr = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if (i-r)**2+(j-r)**2 > n**2*epsilon**2:
                arr[i,j] = 1
    return arr

def Get_V(image):
    '''
    Input:
        image: an image array of circular shape (n,n)
    Output:
        V: the visibility reading from the image. range 0~1.
    '''
    N = len(image) # the side length of the image.
    image = np.multiply(image, Circular_Mask(N)) # apply the aperture mask
    
    fft_img = fftshift(fft2(image))
    k = fftshift(fftfreq(N))
    kx, ky = np.meshgrid(k,k)
    power = np.square(abs(fft_img)) #get power spectrum
    mask = Zero_f_mask(N) #create the 0-f mask
    power = np.multiply(power,mask)
    
    #take the angle from the left hand plane of the image, 
    #then do a rotation of the original image
    half_plane = power[:,int(N/2):]
    kx_half,ky_half = kx[:,int(N/2):],ky[:,int(N/2):]
    ind = np.unravel_index(np.argmax(half_plane, axis=None), half_plane.shape)
    rotation_angle = np.arctan(-ky_half[ind]/kx_half[ind])
    image = PIL.Image.fromarray(image.astype('uint8'),'L')
    image = image.rotate(-rotation_angle/(2*np.pi)*360)
    rotated_arr = np.array(image)
    
    #calculate visibility from here
    #sum intensity over all pixel columns
    I = np.zeros(N)
    # count number of valid pixels over all pixel columns
    N_pixels = np.zeros(N)
    for i in range(N):
        I[i] = np.sum(rotated_arr[:,i])
        N_pixels[i] = np.sum(Circular_Mask(N)[:,i])
    I_adjusted = I[1:]/N_pixels[1:]
    V = (max(I_adjusted)-min(I_adjusted))/(max(I_adjusted)+min(I_adjusted))
    
    return V


        
    