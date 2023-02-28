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
while True:
    img = Get_image(cap)
    keypressed = cv2.waitKey(30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    print(Cal_Visib(img))
    cv2.putText(img, str(round(Cal_Visib(img),4)), (32,64), font, 0.4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('test', img)
    if keypressed == ord('q'):
        break
   