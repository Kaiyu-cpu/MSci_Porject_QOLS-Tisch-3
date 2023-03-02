#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 12:26:51 2023

@author: fanchao
"""

import cv2
from fringe_analysis import Cal_Visib
from camera_reader import Get_image

cap = cv2.VideoCapture(0)

count = 0
font = cv2.FONT_HERSHEY_SIMPLEX
visib = '0'
while True:

    keypressed = cv2.waitKey(30)
    ret,frame = cap.read()
    if count  == 30:
        img = Get_image(cap)
        visib = str(round(Cal_Visib(img),4)) 
        count = 1
    else:
        count += 1
        
    cv2.putText(frame, visib, 
                (32,64), font, 2, (255, 255, 255), 2, cv2.LINE_AA)    
    cv2.imshow('test', frame)
    if keypressed == ord('q'):
        break
   
    
