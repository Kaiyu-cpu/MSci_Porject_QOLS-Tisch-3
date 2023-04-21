#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 12:26:51 2023

@author: fanchao
"""

import cv2
from fringe_analysis import Cal_Visib
from camera_reader import Get_image
from KPZ101 import Initialise,Set_V
def action(Volt):
    '''
    function to adjust the voltages of the kpz piezos
    Input: Volt - type list: the input voltages of 4 mounts
    
    '''
    for i in range (4):
        Set_V(devices[i],Volt[i])
    #time.sleep(sleep_time)
#%%

cap = cv2.VideoCapture(1)
#Serial numbers
SN1="29500948" #M V
SN2="29500732" #M H
SN3="29501050" #BS V
SN4="29500798" #BS H

Serial_num = [SN1, SN2, SN3, SN4]


#set up devices
devices = []
for i in Serial_num:
    devices.append(Initialise(i))
    


#%%
count = 0
font = cv2.FONT_HERSHEY_SIMPLEX
global visib
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


import tkinter as tk

# Define your function that interacts with user input
def do_something_with_input(input_value):
    # Do something with the user input
    #V_list = [int(num) for num in input_value.split(',')]
    action(input_value)

# Define your loop function
def loop_function(visib):
    #cap = cv2.VideoCapture(1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    count = 1
    top_left_x, top_left_y = 100, 100
    bottom_right_x, bottom_right_y = 400, 400

    # Create a tkinter window
    window = tk.Tk()
    '''
    # Create a label and textbox for user input
    label = tk.Label(window, text="Enter a value:")
    label.pack()
    textbox = tk.Entry(window)
    textbox.pack()
    '''
  

    # Create 4 slider bars for user input
    slider1 = tk.Scale(window, from_=0, to=150, orient=tk.HORIZONTAL, label="M2  Vetical")
    slider2 = tk.Scale(window, from_=0, to=150, orient=tk.HORIZONTAL, label="M2 Horizontal")
    slider3 = tk.Scale(window, from_=0, to=150, orient=tk.HORIZONTAL, label="BS2 Vetical")
    slider4 = tk.Scale(window, from_=0, to=150, orient=tk.HORIZONTAL, label="BS2 Horizontal")
    slider1.pack()
    slider2.pack()
    slider3.pack()
    slider4.pack()
    
    # Create a button to submit the user input
    def submit_input():
        input_values = [slider1.get(), slider2.get(), slider3.get(), slider4.get()]
        do_something_with_input(input_values)
    submit_button = tk.Button(window, text="Submit", command=submit_input)
    submit_button.pack()

    # Run the loop
    while True:
        keypressed = cv2.waitKey(30)
        ret,frame = cap.read()
        if count  == 30:
            img = Get_image(cap)
            visib = str(round(Cal_Visib(img),4)) 
           # arr.append(visib)
            count = 1
        else:
            count += 1

        cv2.putText(frame, visib, 
                    (32,64), font, 2, (255, 255, 255), 2, cv2.LINE_AA)  
        img = cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
        cv2.imshow('test', frame)

        # Check if user has pressed 'q' key or closed the window
        if keypressed == ord('q') or cv2.getWindowProperty('test', cv2.WND_PROP_VISIBLE) < 1:
            break

        # Update the tkinter window
        window.update()

    # Release the camera and destroy the tkinter window
    cap.release()
    cv2.destroyAllWindows()
    window.destroy()

# Call your loop function
loop_function(visib = visib)