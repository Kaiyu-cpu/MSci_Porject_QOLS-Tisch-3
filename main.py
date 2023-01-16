# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 22:41:02 2023

@author: Owen
"""

import random
import multiprocessing
from camera_reader import Get_image
from fringe_analysis import Cal_Visib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from KPZ101 import Initialise, Set_V, Kill
from Genetic_Algorithm import GA
#%%
def linear(pop):
    scores=[]
    for i in range (len(pop)):
        scores.append(sum(pop[i]))
    return scores

def test_func(x1,x2):
    return (np.sin(np.pi*x1/86-2*np.pi*x2/51)**2+np.sin(np.pi*x2/102+2*np.pi*x1/43)**2)/(np.sqrt((x1-43)**2+(x2-51)**2)/10+1)

def non_linear(pop):
    scores=[]
    for i in range (len(pop)):
        x1=pop[i][0]
        x2=pop[i][1]
        x3=pop[i][2]
        x4=pop[i][3]
        y = -x3 -x4 +test_func(x1, x2) 
        scores.append(y)
    return scores

def get_Visib(pop):
    Visib_list=np.zeros(len(pop))
    for i in range(len(pop)):
        action(pop[i])
        image=Get_image(cap,1,t_delay=0)[0]
        Visib_list[i]=Cal_Visib(image)
    return Visib_list

def action(Volt):
    if __name__ ==  '__main__':   
        p0=multiprocessing.Process(target=Set_V(devices[0],Volt[0]))
        p1=multiprocessing.Process(target=Set_V(devices[1],Volt[1]))
        p2=multiprocessing.Process(target=Set_V(devices[2],Volt[2]))
        p3=multiprocessing.Process(target=Set_V(devices[3],Volt[3]))
    
        p0.start()
        p1.start()
        p2.start()
        p3.start()
        
        p0.join()
        p1.join()
        p2.join()
        p3.join()
    return

#%%
#Serial numbers
SN1="29500948" #M V
SN2="29500732" #M H
SN3="29501050" #BS V
SN4="29500798" #BS H

Serial_num = [SN1, SN2, SN3, SN4]


#set up the camera
cap = cv2.VideoCapture(0)

#set up devices
devices = []
for i in Serial_num:
    devices.append(Initialise(i))

#%%

# initial population of random dna
n_pop = 16
n_gene = 4
pop = [random.sample(range(0,150), n_gene) for _ in range(n_pop)]

# GA is used to MAXIMISE the target function
scores=GA(pop,target=non_linear,n_iter=1000,tournament_size=3)
  
plt.plot(scores)
plt.xlabel('Number of Iterations')
plt.ylabel('Score')
plt.grid()

# find the stop point 
temp=0
stop=0
for i in range (1,len(scores)):
    if scores[i]!=scores[i-1]:
        temp=0
    if temp>=100:
        stop=i
        break
    temp+=1
plt.plot(stop,scores[stop],'o',color='r')
plt.show()

#%% 3D plot of test function

plt.rcParams["figure.figsize"] = [10, 10]
plt.rcParams["figure.autolayout"] = True

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

xx1 = np.array(np.linspace(0, 150, 1501))
xx2 = np.array(np.linspace(0, 150, 1501))
x1,x2=np.meshgrid(xx1,xx2)

y=test_func(x1,x2)

ax.plot_surface(x1, x2, y, cmap="plasma")

#find global max
Max=0
for i in range (len(xx1)):
    if max(y[i])>Max:
        Max=max(y[i])
        




