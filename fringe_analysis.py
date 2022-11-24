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

#%%
def Fourier_Transform(image):
    fft_image = fft2(image)
    return fft_image

def Beam(x,y,x0,y0,std,E0):
    return (E0*np.exp(-((x-x0)**2+(y-y0)**2)/(2*std**2)))**2

def Interference(Beam1,Beam2,delta):
    return Beam1+Beam2+2*np.sqrt(Beam1*Beam2)*np.cos(delta)
    
    
def phase_diff(x,y,x0,y0,x1,y1,lambda_):
    return 2*np.pi/lambda_*(np.sqrt((x-x0)**2+(y-y0)**2)-np.sqrt((x-x1)**2+(y-y1)**2))

    
x = np.linspace(0,999,1000)
y = np.linspace(0,999,1000)
x, y = np.meshgrid(x,y)
z1 = Beam(x,y,700,300,200,100)
z2 = Beam(x,y,700,300,200,100)
delta = phase_diff(x,y,700,300,700,300,150)

plt.imshow(Interference(z1,z2,delta))




#%%


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

#%%
trial_img = Interference(z1,z2,delta)
fft_trial = Fourier_Transform(trial_img)
fft_trial = fftshift(fft_trial)
fig = plt.figure()
ax = ax = fig.add_axes((0,0,10,10),projection='3d')
ax.plot_surface(x,y,np.log10(abs(fft_trial)),cmap=cm.coolwarm)
plt.show()
