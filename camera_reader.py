import cv2
import numpy as np
import time


def Get_image(camera):
    '''
    the function that returns an image array
    params:
    camera | opencv camera object
    returns:
    data: arrays of images
    '''
     
    
    ret, frame = camera.read()

    print("image shot")
    data = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    data = np.array(data)
    data = cv2.resize(data,(64,64)) #here, need to think about whether to directly reshape
                                        # or crop the image to aviod any distortion
                                        #investigate this with the new camera
   
    
    return data

