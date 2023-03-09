# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 22:41:02 2023

@author: Owen
"""
import random
from camera_reader import Get_image
from fringe_analysis import Cal_Visib
import cv2
import numpy as np
from KPZ101 import Initialise, Set_V, Kill
from Genetic_Algorithm import GA
import matplotlib.pyplot as plt
import time
from bayes_opt import BayesianOptimization
import pandas as pd
import PIL
import pygad
import datetime

#%% define useful functions here

def get_Visib(pop):
    Visib_list=np.zeros(len(pop))
    for i in range(len(pop)):
        action(pop[i])
        image=Get_image(cap)
        Visib_list[i]=Cal_Visib(image)
    return Visib_list

def action(Volt):
    '''
    function to adjust the voltages of the kpz piezos
    Input: Volt - type list: the input voltages of 4 mounts
    
    '''
    for i in range (4):
        Set_V(devices[i],Volt[i])
    time.sleep(0.5)
    

def fitness_func(pop):
    Visib_list = np.array(get_Visib(pop))
    fitness_list = Visib_list - np.log(1-Visib_list) - 1
    return fitness_list


def worst_start(camera):
    V_list = []
    Visib_list = []
    for i in range(20):
        V_arr = np.zeros(4)
        for j in range(4):
            V_arr[j] = random.random()*150
        V_list.append(V_arr)
        action(V_arr)
        img = Get_image(camera)
        Visib_list.append(Cal_Visib(img))
    index = np.where(Visib_list == np.min(Visib_list))[0][0]
    return V_list[index]

#%% Initilise K-Cubes and camera

#set up the camera
cap = cv2.VideoCapture(1)

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


#%% Bayesian module

def Bayes_fitness(V1,V2,V3,V4):
    V = np.array([V1,V2,V3,V4])
    action(V)
    image=Get_image(cap)
    visib = Cal_Visib(image)
    return visib - np.log10(1-visib)



# Bounded region of parameter space
pbounds = {'V1':(0,150),'V2':(0,150),'V3':(0,150),'V4':(0,150)}



Big_array = []
date_string = str(datetime.datetime.now())
for i in range(8):
    optimizer = BayesianOptimization(
        f=Bayes_fitness,
        pbounds=pbounds,
        random_state=0,
        allow_duplicate_points=True)
    v_initial = []
    for j in range(4):
        v_initial.append(random.random()*150)
    action(v_initial) #get a random starting point
    im_before = Get_image(cap)
    im = PIL.Image.fromarray(im_before.astype('uint8'),'L')
    im.save('initial image {}.jpg'.format(i))
    optimizer.maximize(init_points=1,n_iter=99)
    V1 = optimizer.max['params']['V1']
    V2 = optimizer.max['params']['V2']
    V3 = optimizer.max['params']['V3']
    V4 = optimizer.max['params']['V4']
    action([V1,V2,V3,V4])
    im_after = Get_image(cap)
    im = PIL.Image.fromarray(im_after.astype('uint8'),'L')
    im.save('final image {}.jpg'.format(i))
    records = optimizer.space.target
    Big_array.append(records)
    
    
df = pd.DataFrame(Big_array)   

df.to_csv(date_string+'\Bayesian record'+date_string+'.csv')    



#%% PyGad module
#define parameters
gene_space = {'low':0, 'high':150}

def pygad_fitness(V, V_idx):
    action(V)
    image=Get_image(cap)
    visib = Cal_Visib(image)
    return visib - np.log10(1-visib)


fitness_function = pygad_fitness

num_generations = 100
num_parents_mating = 4

sol_per_pop = 8
num_genes = 4

init_range_low = 0
init_range_high = 150

parent_selection_type = "rank"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 25
date_string = str(datetime.datetime.now())
date_string = date_string.replace(':','_')
date_string = date_string.replace('.','_')

Big_array=[]
for i in range(8):
    v_initial = worst_start(cap)
    action(v_initial) #get a random starting point
    im_before = Get_image(cap)
    #plt.imshow(im_before,cmap='gray')
    im = PIL.Image.fromarray(im_before.astype('uint8'),'L')
    im.save('initial image pygad{}.jpg'.format(i))
    
    
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           gene_space = gene_space,
                           gene_type = float,
                           save_solutions=True,
                           keep_elitism=0)
    
    ga_instance.run()
    ga_instance.plot_fitness()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    action(solution)
    im_after = Get_image(cap)
    im = PIL.Image.fromarray(im_after.astype('uint8'),'L')
    im.save('final image pygad{}.jpg'.format(i))
    
        
    results = ga_instance.solutions_fitness
    Big_array.append(results)

df = pd.DataFrame(Big_array)   

df.to_csv('\PyGAD record'+date_string+'.csv') 
 
#%% this cell shuts down all devices
for i in range (4):
    Kill(devices[i])