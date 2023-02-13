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


#%% define useful functions here

def get_Visib(pop):
    Visib_list=np.zeros(len(pop))
    for i in range(len(pop)):
        action(pop[i])
        image=Get_image(cap)
        global count
        pop_num=count%8
        iteration_num=count//8
        count += 1
        #cv2.imwrite(f'iteration {iteration_num}, population {pop_num}.png',image)
        Visib_list[i]=Cal_Visib(image)
    return Visib_list

def action(Volt):
    '''
    function to adjust the voltages of the kpz piezos
    Input: Volt - type list: the input voltages of 4 mounts
    
    '''
    for i in range (4):
        Set_V(devices[i],Volt[i])
    

def fitness_func(pop):
    Visib_list = np.array(get_Visib(pop))
    fitness_list = Visib_list - np.log(1-Visib_list) - 1
    return fitness_list




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

#%% Run GA get scores

count = 0

# initial population of random dna
n_pop = 8
n_gene = 4
pop = [random.sample(range(0,75), n_gene) for _ in range(n_pop)]

n_run=1
n_iteration=500
scores_ensemble=[]

start=time.time()


im_before = Get_image(cap)



for i in range (n_run):
    # GA is used to MAXIMISE the target function
    scores=GA(pop,target=fitness_func,n_iter=n_iteration,tournament_size=3)
    scores_ensemble.append(scores)

end=time.time()
t=end-start

im_after = Get_image(cap)

print(f'time taken for {n_iteration}iter is {t}')
scores_ensemble=np.array(scores_ensemble)
ensemble_mean=scores_ensemble.mean(axis=0)

#%% Bayesian module
from bayes_opt import BayesianOptimization



def Bayes_fitness(V1,V2,V3,V4):
    V = np.array([V1,V2,V3,V4])
    action(V)
    image=Get_image(cap)
    visib = Cal_Visib(image)
    return visib - np.log(1-visib) - 1



# Bounded region of parameter space
pbounds = {'V1':(0,75),'V2':(0,75),'V3':(0,75),'V4':(0,75)}





optimizer = BayesianOptimization(
    f=Bayes_fitness,
    pbounds=pbounds,
    random_state=1,
    allow_duplicate_points=True)


optimizer.maximize(init_points=1,n_iter=30)

plt.plot(range(1, 1 + len(optimizer.space.target)), optimizer.space.target, "-o")


#%% PyGad module

#define parameters
gene_space = {'low':0, 'high':75}


def pygad_fitness(V, V_idx):
    action(V)
    image=Get_image(cap)
    visib = Cal_Visib(image)
    return visib - np.log(1-visib) - 1


fitness_function = pygad_fitness

num_generations = 10
num_parents_mating = 4

sol_per_pop = 8
num_genes = 4

init_range_low = 0
init_range_high = 75

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 12.5

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
                       gene_type = float)


#%% this cell shuts down all devices
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
'''
below are temporarily useless codes saved up for later
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



