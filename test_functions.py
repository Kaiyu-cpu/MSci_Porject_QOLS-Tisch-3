# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:17:12 2023

@author: chaof
"""
import numpy as np
import matplotlib.pyplot as plt


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

#%% 3D plot of test function

#below are temporarily useless codes saved up for later
plt.rcParams["figure.figsize"] = [12, 12]
plt.rcParams["figure.autolayout"] = True

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

xx1 = np.array(np.linspace(0, 100, 1501))
xx2 = np.array(np.linspace(0, 100, 1501))
x1,x2=np.meshgrid(xx1,xx2)

y=test_func(x1,x2)

ax.plot_surface(x1, x2, y, cmap="coolwarm")

ax.set_xlabel('x1',fontsize=20)
ax.set_ylabel('x2',fontsize=20)
ax.set_zlabel('objective',fontsize=20)
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

