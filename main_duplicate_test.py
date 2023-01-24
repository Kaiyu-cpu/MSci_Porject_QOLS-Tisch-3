# -*- coding: utf-8 -*-
"""
This is a duplicate file of the main script to test the new functions, 
in real ussage use the main file
"""
import random
import multiprocessing
from camera_reader import Get_image
from fringe_analysis import Cal_Visib
import cv2
import numpy as np
from KPZ101 import Initialise, Set_V, Kill
import pygad
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
        image=Get_image(cap,want_pic=False)
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

#%%
#define parameters


function_inputs = [0,0,0,0]
desired_output = 300


gene_space = {'low':0, 'high':75}

def fitness_func(solution, solution_idx):
    output = -solution[0]-solution[1]+test_func(solution[2],solution[3])
    fitness = 1.0 / np.abs(output - desired_output)
    return output


fitness_function = fitness_func

num_generations = 500
num_parents_mating = 4

sol_per_pop = 8
num_genes = len(function_inputs)

init_range_low = 0
init_range_high = 75

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10


#%%
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
                       gene_space = gene_space)
    
ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

prediction = np.sum(np.array(function_inputs)*solution)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

    
    
ga_instance.plot_fitness()
    
#%% this cell shuts down all devices
for i in range (4):
    Kill(devices[i])

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



