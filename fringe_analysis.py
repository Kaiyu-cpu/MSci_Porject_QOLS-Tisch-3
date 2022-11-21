# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 20:39:39 2022

@author: chaof
"""
import numpy as np
from scipy.fft import fft2, ifft2
import matplotlib.pyplot as plt
import cv2
import os
plt.figure(figsize=(20, 12),dpi=200) 


def Fourier_Transform(image):
    fft_image = fft2(image)
    return fft_image

def plot_spectrum(im_fft):
    # A logarithmic colormap
    plt.imshow(np.log(np.abs(im_fft)))
    plt.colorbar()

folder = 'fringes&visib/'  #paths to folder containing images
images = []
for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    if img is not None:
        images.append(img)

count = [1,3,5,7,9,11]        
for i in range(6):
    image = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
    fft_image = Fourier_Transform(image)
    plt.subplot(6,2,count[i])
    plt.imshow(image)
    plt.subplot(6,2,count[i]+1)
    plot_spectrum(fft_image)
plt.tight_layout()
plt.show()
