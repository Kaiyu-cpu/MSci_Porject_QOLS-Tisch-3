# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 17:00:56 2022

@author: chaof
"""
import random
import subprocess
import multiprocessing
from camera_reader import Get_image
from fringe_analysis import Cal_V
import cv2
import numpy as np
import time

#%%


Volt=[1,2,3,4]

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

#%%

start=time.perf_counter()

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

finish=time.perf_counter()
print(finish-start)
    
#%%  
start=time.perf_counter()
change_V(SN1,"10")
change_V(SN2,"20")
change_V(SN3,"30")
change_V(SN4,"40")
finish=time.perf_counter()
print(finish-start)










    