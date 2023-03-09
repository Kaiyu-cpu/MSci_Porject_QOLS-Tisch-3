#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 12:26:51 2023

@author: fanchao
"""

import cv2
from fringe_analysis import Cal_Visib
from camera_reader import Get_image

cap = cv2.VideoCapture(1)

count = 0
font = cv2.FONT_HERSHEY_SIMPLEX
visib = '0'
ret, frame = cap.read()
height, width, channels = frame.shape
center_x = int(width / 2)
center_y = int(height / 2)
box_size = 128

# Calculate the coordinates of the top-left corner of the box
top_left_x = center_x - int(box_size / 2)
top_left_y = center_y - int(box_size / 2)

# Calculate the coordinates of the bottom-right corner of the box
bottom_right_x = center_x + int(box_size / 2)
bottom_right_y = center_y + int(box_size / 2)


arr = []
while True:

    keypressed = cv2.waitKey(30)
    ret,frame = cap.read()
    if count  == 30:
        img = Get_image(cap)
        visib = str(round(Cal_Visib(img),4)) 
        arr.append(visib)
        count = 1
    else:
        count += 1
        
    cv2.putText(frame, visib, 
                (32,64), font, 2, (255, 255, 255), 2, cv2.LINE_AA)  
    img = cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
    cv2.imshow('test', frame)
    if keypressed == ord('q'):
        break
   
    
