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
from KPZ101 import Initialise, Set_V, Kill, ISK
from Genetic_Algorithm import GA
import matplotlib.pyplot as plt
import time
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
    '''
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
    '''
    for i in range (4):
        Set_V(devices[i],Volt[i])
    
    return

#%% Initilise K-Cubes and camera

#set up the camera
cap = cv2.VideoCapture(0)

#Serial numbers
SN1="29500948" #M V
SN2="29500732" #M H
SN3="29501050" #BS V
SN4="29500798" #BS H

Serial_num = [SN1, SN2, SN3, SN4]


#set up devices
devices = []
for i in Serial_num:
    devices.append(Initialise(i))

#%% Run GA get scores

# initial population of random dna
n_pop = 8
n_gene = 4
pop = [random.sample(range(0,75), n_gene) for _ in range(n_pop)]

n_run=1
n_iteration=10
scores_ensemble=[]

start=time.time()

for i in range (n_run):
    # GA is used to MAXIMISE the target function
    scores=GA(pop,target=get_Visib,n_iter=n_iteration,tournament_size=3)
    scores_ensemble.append(scores)

end=time.time()
t=end-start
print(f'time taken for {n_iteration}iter is {t}')
scores_ensemble=np.array(scores_ensemble)
ensemble_mean=scores_ensemble.mean(axis=0)

#%%
for i in range (4):
    Kill(devices[i])

#%% Plot of score vs iteration

stop_list=np.zeros(n_run)

for i in range (n_run):
    plt.plot(scores_ensemble[i],alpha=0.5,linewidth=1)
    # find the stop point 
    temp=0
    stop=0
    for j in range (1,len(scores_ensemble[i])):
        if scores_ensemble[i][j]!=scores_ensemble[i][j-1]:
            temp=0
        if temp>=100:
            stop=j
            stop_list[i]=stop
            break
        temp+=1
    plt.plot(stop,scores_ensemble[i][stop],'.',color='r', alpha=0.7)

plt.xlabel('Number of Iterations')
plt.ylabel('Score')
plt.grid()

plt.plot(ensemble_mean,linewidth=3,label='ensemble mean')

stop_mean=int(np.ceil(np.mean(stop_list))) # rounded up
plt.plot(stop_mean,ensemble_mean[stop_mean],'o',color='r',label='stop point')

end_score=round(ensemble_mean[stop_mean],2)
plt.title(f'non-linear, pop={n_pop}, avg stop point={stop_mean}th iter, avg ending score={end_score}')

plt.legend()

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

plt.show()
        
#%% Calculate stop point vs pop

n_run=20 #number of runs in an ensemble
stop_mean_list=[]
end_score_list=[]

for pop_size in range (4,21,2):
    
    n_gene = 4
    pop = [random.sample(range(0,150), n_gene) for _ in range(pop_size)]
    
    scores_ensemble=[]
    
    for i in range (n_run):
        # GA is used to MAXIMISE the target function
        scores=GA(pop,target=non_linear,n_iter=1000,tournament_size=3)
        scores_ensemble.append(scores)
    
    scores_ensemble=np.array(scores_ensemble)
    ensemble_mean=scores_ensemble.mean(axis=0)
        
    stop_list=np.zeros(n_run)

    for i in range (n_run):
        # find the stop point 
        temp=0
        stop=0
        for j in range (1,len(scores_ensemble[i])):
            if scores_ensemble[i][j]!=scores_ensemble[i][j-1]:
                temp=0
            if temp>=100:
                stop=j
                stop_list[i]=stop
                break
            temp+=1
            
    stop_mean=int(np.ceil(np.mean(stop_list))) # rounded up
    end_score=round(ensemble_mean[stop_mean],2) # rounded to 2 d.p.
    stop_mean_list.append(stop_mean)
    end_score_list.append(end_score)

#%% Plot stop point vs pop

pop_size=[i for i in range(4,21,2)]

# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(pop_size,
        stop_mean_list,
        color="red", 
        marker="o")
# set x-axis label
ax.set_xlabel("population size", fontsize = 14)
# set y-axis label
ax.set_ylabel("average stop point",
              color="red",
              fontsize=14)

# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(pop_size, end_score_list,color="blue",marker="o")
ax2.set_ylabel("average ending score",color="blue",fontsize=14)
plt.title(f'non-linear target function, ensemble size={n_run}')
plt.show()



