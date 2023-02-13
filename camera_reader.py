import cv2
import numpy as np


def Get_image(camera, l = 64):
    '''
    the function that returns an image array
    params:
    camera | opencv camera object
    l | size of cropped image pixel
    returns:
    data: arrays of images
    '''
     
    
    ret, frame = camera.read()

    print("image shot")
    image = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    center = image.shape
    x = center[1]/2 - l/2 
    y = center[0]/2 - l/2 + 10 # to avoid camera defect

    crop_img = image[int(y):int(y+l), int(x):int(x+l)]
    
    return crop_img

