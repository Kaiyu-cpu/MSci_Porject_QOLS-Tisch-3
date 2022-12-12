# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 12:28:11 2022

@author: Owen
"""
import multiprocessing
import time
import subprocess


    
start=time.perf_counter()

def func():
    p=subprocess.run([r"D:\Thorlabs\C#Programs\KPZ101Console\bin\Debug\KPZ101Console.exe","29500948","30"], capture_output=True, text=True)
    print(p.stdout)

if __name__ ==  '__main__':   
    p1=multiprocessing.Process(target=func)
    p2=multiprocessing.Process(target=func)


    p1.start()
    p2.start()
    
    p1.join()
    p2.join()

finish=time.perf_counter()

print(finish-start)