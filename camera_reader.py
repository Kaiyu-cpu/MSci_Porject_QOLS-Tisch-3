import cv2
import numpy as np
#setting up the camera
camera = cv2.VideoCapture(0) #index is a bit chaotic, not sure which one it initiates

def get_image(camera,n,want_pic=False):
    '''
    the function that returns n image tensors
    params:
    camera | opencv camera object
    n | number of images specified
    want_pic | whether you want the picture files to be stored; Default is no
    returns:
    data: arrays of images
    '''
    data = np.zeros(n,dtype=object)
    for i in range(n):
        ret, frame = camera.read()
        data[i] = frame.astype('int32')
        if want_pic == True:
            cv2.imwrite('image{}.png'.format(i), frame)
    return data

#trying it out
x = get_image(camera,10,want_pic=True)
print(x)
