#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 15:23:33 2022

@author: fanchao
"""

#%% import packages
import numpy as np
from numpy.fft import fft2,fftshift,ifftshift,fftfreq
import matplotlib.pyplot as plt
import cv2
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import PIL


#%%
def Fringe(I0,V,f,x,y,phi):
    return I0*(1+V*np.cos(2*np.pi*f*x*np.cos(phi)+2*np.pi*f*y*np.sin(phi)))

def Aperture(d):
    r = d/2
    arr = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            if (i-r)**2+(j-r)**2 < r**2:
                arr[i,j] = 1
    return arr


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

#%%

x = np.linspace(0,99,100)
y = np.linspace(0,99,100)
x, y = np.meshgrid(x,y)
image = Fringe(100,0.8,0.08,x,y,0.8)
image = np.multiply(image,Aperture(100))
plt.rcParams.update({'font.size': 10})
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
#%%
fft_trial = fft2(image)
fft_trial = fftshift(fft_trial)
kx,ky = fftfreq(100),fftfreq(100)
kx,ky = fftshift(kx),fftshift(ky)
kx,ky = np.meshgrid(kx,ky)

power = np.square(abs(fft_trial))

fig = plt.figure()
plt.rcParams.update({'font.size': 80})
ax = fig.add_axes((0,0,10,10),projection='3d')
ax.axis('auto')
ax.plot_surface(kx,ky,power,cmap=cm.coolwarm)
plt.show()

plt.rcParams.update({'font.size': 10})
plt.imshow(power,cmap=cm.coolwarm,extent=[min(kx[0]),max(kx[0]),max(ky[0]),min(ky[-1])])


#%% remove zero frequency component, and convert coordinate

def Zero_f_mask(epsilon,n):
    r = n/2
    arr = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if (i-r)**2+(j-r)**2 > n**2*epsilon**2:
                arr[i,j] = 1
    return arr

mask = Zero_f_mask(0.05, 100)

masked_power = np.multiply(power,mask)
fig = plt.figure()
plt.rcParams.update({'font.size': 80})
ax = fig.add_axes((0,0,10,10),projection='3d')
ax.axis('auto')
ax.plot_surface(kx,ky,masked_power,cmap=cm.coolwarm)
plt.show()
                
plt.rcParams.update({'font.size': 10})
plt.imshow(masked_power,cmap=cm.coolwarm,extent=[min(kx[0]),max(kx[0]),min(ky[0]),max(ky[-1])])


#%% take the angle from the left hand plane of the image, do a rotation of the original image
half_plane = masked_power[:,50:]
kx_half,ky_half = kx[:,50:],ky[:,50:]
plt.imshow(half_plane,cmap=cm.coolwarm,extent=[min(kx[0][50:]),max(kx[0][50:]),min(ky[0]),max(ky[-1])])


ind = np.unravel_index(np.argmax(half_plane, axis=None), half_plane.shape)
print(kx_half[ind],-ky_half[ind])  #note that ky starts from negative values

rotation_angle = np.arctan(-ky_half[ind]/kx_half[ind])

image = PIL.Image.fromarray(image.astype('uint8'),'L')
image = image.rotate(-rotation_angle/(2*np.pi)*360)
plt.imshow(image,cmap='gray')
#%%
rotated_arr = np.array(image)
print(rotated_arr)

plt.imshow(rotated_arr)

#%% calculate visibility from here
#sum intensity over all pixel columns
I = np.zeros(len(rotated_arr))
# count number of valid pixels over all pixel columns
N = np.zeros(len(rotated_arr))
for i in range(len(rotated_arr)):
    I[i] = np.sum(rotated_arr[:,i])
    N[i] = np.sum(Aperture(len(rotated_arr))[:,i])
for i in range(len(N)):
    if N[i] == 0:
        N[i] = 1
plt.plot(I)
plt.xlabel('x\'')
plt.ylabel('Sum of Intensities at each pixel on the row')
plt.show()
plt.plot(N)
plt.xlabel('x\'')
plt.ylabel('number of valid pixel points')
plt.show()
I_adjusted = I/N
I_adjusted = I_adjusted[1:]
plt.plot(I_adjusted)
plt.xlabel('x\'')
plt.ylabel('Mean intensity for each row')
plt.show()

V = (max(I_adjusted)-min(I_adjusted))/(max(I_adjusted)+min(I_adjusted))
print(V)