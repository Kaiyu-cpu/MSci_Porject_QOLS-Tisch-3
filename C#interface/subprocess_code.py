# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 11:05:49 2022

@author: Owen
"""
import subprocess
import time


start=time.perf_counter()

#p1=subprocess.run('dir',shell=True)
#print(p1)

p2 = subprocess.run([r"D:\Thorlabs\C#Programs\KPZ101Console\bin\Debug\KPZ101Console.exe","29500948","30"], capture_output=True, text=True)
print(p2.stdout)

#p3 = subprocess.run([r"D:\Thorlabs\C#Programs\KPZ101Console\bin\Debug\KPZ101Console.exe","29500949","30"], capture_output=True, text=True)
#print(p3.stdout)

finish=time.perf_counter()

print(finish-start)