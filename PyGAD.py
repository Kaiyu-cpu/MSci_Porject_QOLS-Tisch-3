#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 18:56:33 2023

@author: fanchao
"""

import pygad
import numpy as np

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
    
    
    
    
    
    
    
    
    
    
    
    
    