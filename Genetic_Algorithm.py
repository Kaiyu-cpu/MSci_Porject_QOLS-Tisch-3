# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 21:47:46 2023

@author: Owen
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from camera_reader import Get_image

def GA(pop, target, n_iter=1, p_cross=0.9, p_mut=0.1, tournament_size=3, Print=False):
    
    n_pop = len(pop) 
    scores_best = np.zeros(n_iter)
    improvement=0.0
    
    for gen in range (n_iter):
        
        # evaluate all candidates in the population
        scores = list(target(pop))
        
        #append scores array to plot model performance
        scores_best[gen] = max(scores)
        #if gen % 10 == 0 and gen!=0:
            #print('Improvement after 10 iterations: ',max(scores)-improvement)
            #improvement = max(scores)
            
        # check for new best solution
        best_score=max(scores)
        best_index=scores.index(best_score)
        best_dna=pop[best_index]
        
        if Print==True: #print generation, the best dna and its sorce
            print(">%d, new best f(%s) = %.3f" % (gen,  best_dna, best_score))
        
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
         
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, p_cross):
                # mutation
                children.append(mutation(c, p_mut))
        
        # replace population
        pop = children
        
    return scores_best

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
            #dna[i] = random.randint(0,75)
            dna[i] = random.random()*75
    return dna