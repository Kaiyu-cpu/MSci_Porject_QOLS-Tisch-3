# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 20:34:23 2022

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
#%%
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
    
# tournament selection
def selection(pop, scores, k=3):
    # first random selection
    selected_index = random.randint(0,len(pop)-1)
    index=list(range(len(pop)))
    index.remove(selected_index)
    for i in random.sample(index,k-1):
        # check if better (e.g. perform a tournament)
        if scores[i] > scores[selected_index]:
            selected_index = i
    return pop[selected_index]


# crossover two parents to create two children
def crossover(p1, p2, p_cross): #p_cross is the prob of crossover
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if random.randint(0,99) < p_cross*100:
        # select crossover point that is not on the end of the string
        pt = random.randint(1, len(p1)-2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


# mutation operator
def mutation(dna, p_mut): #p_mut is the prob of mutation
    for i in range(len(dna)):
        # check for a mutation
        if random.randint(0,99) < p_mut*100:
            # mutate the gene
            dna[i] = random.randint(0,150)
    return dna

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
n_pop = 4

n_gene = 4

pop = [random.sample(range(0,150), n_gene) for _ in range(n_pop)]

# keep track of best solution
best_dna, best_score = 0, 0

# enumerate generations
n_iter=1

scores_list = np.zeros(n_iter)

# print improvements once in a while
improvement = 0.0

for gen in range(n_iter):

    # evaluate all candidates in the population
    scores = list(get_Visib(pop))
    
    #append scores array to plot model performance
    scores_list[gen] = max(scores)
    #if gen % 10 == 0:
        #print('Improvement after 10 iterations: ',max(scores)-improvement)
        #improvement = max(scores)
        
    # check for new best solution
    best_score=max(scores)
    best_index=scores.index(best_score)
    best_dna=pop[best_index]
    print(">%d, new best f(%s) = %.3f" % (gen,  best_dna, best_score))
    
    # select parents
    selected = [selection(pop, scores) for _ in range(n_pop)]
     
    # create the next generation
    children = list()
    for i in range(0, n_pop, 2):
        # get selected parents in pairs
        p1, p2 = selected[i], selected[i+1]
        # crossover and mutation
        p_cross=0.9
        p_mut=0.1
        for c in crossover(p1, p2, p_cross):
            # mutation
            children.append(mutation(c, p_mut))
    
    # replace population
    pop = children

for i in devices: #shut down all devices
    Kill(i)    
    
plt.plot(scores_list,'o')
plt.xlabel('Number of Iterations')
plt.ylabel('Visibility')
plt.grid()
plt.show()
