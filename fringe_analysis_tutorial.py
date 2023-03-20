#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:56:31 2023

@author: fanchao
"""

#%% import packages
import numpy as np
from numpy.fft import fft2,fftshift,fftfreq
import matplotlib.pyplot as plt
from matplotlib import cm
import PIL
import cv2

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

#%% test using a real image
image = cv2.imread("/Users/fanchao/Desktop/Imperial/Year 4/Msci_Project/Plots and graphs/V.png")
l = 128
center = image.shape
x = center[1]/2 - l/2 
y = center[0]/2 - l/2 + 10 # to avoid camera defect

crop_img = image[int(y):int(y+l), int(x):int(x+l)]

crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
crop_img = np.multiply(crop_img, Aperture(128))

plt.imshow(crop_img,cmap='gray')
#plt.imshow(crop_img[:,:,[2,1,0]])
plt.axis('off')
#plt.title('Original Image from Camera')

#%%
image = crop_img
fft_trial = fft2(image)
fft_trial = fftshift(fft_trial)
kx,ky = fftfreq(128),fftfreq(128)
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

mask = Zero_f_mask(0.05, 128)

masked_power = np.multiply(power,mask)
fig = plt.figure()
plt.rcParams.update({'font.size': 80})
ax = fig.add_axes((0,0,10,10),projection='3d')
ax.axis('auto')
ax.plot_surface(kx,ky,masked_power,cmap=cm.coolwarm)
plt.show()
                
plt.rcParams.update({'font.size': 10})
plt.imshow(masked_power,cmap=cm.coolwarm,extent=[min(kx[0]),max(kx[0]),min(ky[0]),max(ky[-1])])
plt.axis('off')



#%% take the angle from the left hand plane of the image, do a rotation of the original image
half_plane = masked_power[:,64:]
kx_half,ky_half = kx[:,64:],ky[:,64:]
plt.imshow(half_plane,cmap=cm.coolwarm,extent=[min(kx[0][64:]),max(kx[0][64:]),min(ky[0]),max(ky[-1])])


ind = np.unravel_index(np.argmax(half_plane, axis=None), half_plane.shape)
print(kx_half[ind],-ky_half[ind])  #note that ky starts from negative values

rotation_angle = np.arctan(-ky_half[ind]/kx_half[ind])

image = PIL.Image.fromarray(image.astype('uint8'),'L')
image = image.rotate(-rotation_angle/(2*np.pi)*360-0.5)

plt.imshow(image,cmap='gray')
plt.axis('off')

#%% calculate visibility from here
#sum intensity over all pixel columns
rotated_arr = np.array(image)
I = np.zeros(len(rotated_arr))
# count number of valid pixels over all pixel columns
N = np.zeros(len(rotated_arr))
for i in range(len(rotated_arr)):
    I[i] = np.sum(rotated_arr[:,i])
    N[i] = np.sum(Aperture(len(rotated_arr))[:,i])
for i in range(len(N)):
    if N[i] == 0:
        N[i] = 1

#%%
plt.figure(dpi=1500)
plt.rcParams["figure.figsize"] = (12,8)
plt.rcParams.update({'font.size': 22})
plt.plot(I)
plt.xlabel('column index')
plt.ylabel('Sum of $I$ over one column')
plt.grid()
plt.show()
plt.plot(N)
plt.xlabel('column index')
plt.ylabel('number of valid pixel points')
plt.grid()
plt.show()
I_adjusted = I/N
I_adjusted = I_adjusted[1:]
plt.plot(I_adjusted)
plt.xlabel('column index')
plt.ylabel('$I_{mean}$ over one column')
plt.xlim((0,127))
plt.grid()
plt.show()

V = (max(I_adjusted)-min(I_adjusted))/(max(I_adjusted)+min(I_adjusted))
print(V)

#%% new way of calculating the visibility: use curve_fit
from scipy.optimize import minimize, curve_fit
def Sin_squared(x,A,w,phi,I0):
    '''
    A: amplitude of square sin wave (Imax)
    w: omega, angular frequency
    phi: offset phase angle
    I0: offset of curve (Imin)
    '''
    return A*np.sin(w*x+phi)**2+I0

x = np.linspace(0,126,127)
guess = np.array([150,0.2,0,50])
popt,pcov = curve_fit(Sin_squared,x,I_adjusted,guess,bounds=((0,0,0,0),(255,np.inf,2*np.pi,255)))

def min_function(params, x, y):
    model = Sin_squared(x, *params)
    residual = ((y - model) ** 2).sum()
    
    if np.any(model > 255):
        residual += 100  # Just some large value
    if np.any(model < 0):
        residual += 100

    return residual

res = minimize(min_function, x0=guess, args=(x, I_adjusted))



plt.plot(x,Sin_squared(x,*popt)-20)
#plt.plot(x,Sin_squared(x,*res.x))
plt.plot(I_adjusted)

#%%
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
I_smooth=savgol_filter(I_adjusted,5,3)
plt.plot(I_smooth, label='smoothed data')

I_max_indices=list(argrelextrema(I_smooth, np.greater)[0])
I_min_indices=list(argrelextrema(I_smooth, np.less)[0])

I_max=[]
I_min=[]

I_max_remove=[]
I_min_remove=[]

I_mean=np.mean(I_smooth)
for i in I_max_indices:
    if(I_smooth[i]>I_mean):
        I_max.append(I_smooth[i])
    else:
        I_max_remove.append(i)
for i in I_min_indices:
    if(I_smooth[i]<I_mean):
        I_min.append(I_smooth[i])
    else:
        I_min_remove.append(i)
        
for i in I_max_remove:
    I_max_indices.remove(i)
    
for i in I_min_remove:
    I_min_indices.remove(i)

plt.scatter(I_max_indices,I_max, s = 50, color = 'red',label='maxima')
plt.scatter(I_min_indices,I_min, s = 50, color = 'orange', label ='minima')
I_max_mean=np.mean(I_max)
I_min_mean=np.mean(I_min)
plt.ylim(50,255)
plt.legend()
plt.grid()
plt.xlabel('column index')
plt.ylabel('$I_{mean}$ over one column')
V=(I_max_mean-I_min_mean)/(I_max_mean+I_min_mean)
print(V)






















