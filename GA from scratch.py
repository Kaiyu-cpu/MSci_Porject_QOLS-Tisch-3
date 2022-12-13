# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 20:34:23 2022

@author: Owen
"""
import random
import subprocess
import multiprocessing
from camera_reader import Get_image
from fringe_analysis import Cal_V
import cv2
import numpy as np
import time




def objective (x): #just for test
    return sum(x)

def get_V(pop):
    V_list=np.zeros(len(pop))
    for i in range(len(pop)):
        action(pop[i])
        image=Get_image(cap,1,t_delay=0)[0]
        V_list[i]=Cal_V(image)
    return V_list

def action(Volt):
    def change_V(No,V):
        subprocess.run(
            [r"C:\Users\chaof\Documents\GitHub\MSci_Project_QOLS-Tisch-3\KPZ101Console\bin\Debug\KPZ101Console.exe",No,V], capture_output=True, text=True)
        #print(p.stdout)
    V1=str(Volt[0])
    V2=str(Volt[1])
    V3=str(Volt[2])
    V4=str(Volt[3])
    SN1="29500948" #M V
    SN2="29500732" #M H
    SN3="29501050" #BS V
    SN4="29500798" #BS H
    if __name__ ==  '__main__':   
        p1=multiprocessing.Process(target=change_V(SN1,V1))
        p2=multiprocessing.Process(target=change_V(SN2,V2))
        p3=multiprocessing.Process(target=change_V(SN3,V3))
        p4=multiprocessing.Process(target=change_V(SN4,V4))
    
        p1.start()
        p2.start()
        p3.start()
        p4.start()
        
        p1.join()
        p2.join()
        p3.join()
        p4.join()
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


#setting up the camera
cap = cv2.VideoCapture(0)

#while True:

    #ret, frame = cap.read()
   # cv2.imshow('frame',frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

#cap.release()
#cv2.destroyAllWindows()




# initial population of random dna
n_pop=4

n_gene=4

pop = [random.sample(range(0,150), n_gene) for _ in range(n_pop)]

# keep track of best solution
best_dna, best_score = 0, 0

# enumerate generations
n_iter=1

for gen in range(n_iter):

    # evaluate all candidates in the population
    scores = list(get_V(pop))
    
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
    
    