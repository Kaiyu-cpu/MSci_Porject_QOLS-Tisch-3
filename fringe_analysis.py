# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 20:39:39 2022

@author: chaof
"""
import numpy as np
from scipy.fft import fft2,fftshift
import matplotlib.pyplot as plt
import cv2
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def Fourier_Transform(image):
    fft_image = fft2(image)
    return fft_image

def plot_spectrum(im_fft,ax):
    # A logarithmic colormap
    ax.imshow(np.log10(np.abs(im_fft)))
    #plt.colorbar()

folder = 'fringes&visib/'  #paths to folder containing images
images = []
for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    if img is not None:
        images.append(img)

fig = plt.figure(figsize=(20, 12),dpi=200) 
count = [1,4,7,10,13,16]        
for i in range(6):
    image = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
    fft_image = Fourier_Transform(image)
    fft_image = fftshift(fft_image)
    ax = fig.add_subplot(6,3,count[i])
    ax.imshow(image)
    ax = fig.add_subplot(6,3,count[i]+1)
    plot_spectrum(fft_image,ax)
    ax = fig.add_subplot(6,3,count[i]+2,projection='3d')
    nx, ny = fft_image.shape
    x = np.linspace(0,nx-1,nx)
    y = np.linspace(0,ny-1,ny)
    x, y = np.meshgrid(x,y)
    z = np.log10(np.abs(fft_image)).T
    ax.plot_surface(x, y, z, cmap = cm.coolwarm)
    
plt.tight_layout()
plt.show()

