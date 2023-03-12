import cv2
import numpy as np


def Get_image(camera, center=(480,640), l=128, color=False):
    '''
    the function that returns an image array
    params:
    camera | opencv camera object
    center | center of the image (tuple of integers)
    l | size of cropped image pixel
    returns:
    data: arrays of images
    '''
    ret, frame = camera.read()
    if color == False and len(frame.shape) == 3 and frame.shape[2] == 3:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        image = frame
    x = center[1] // 2 - l // 2
    y = center[0] // 2 - l // 2 + 10  # to avoid camera defect
    crop_img = image[int(y):int(y+l), int(x):int(x+l)]
    return crop_img
