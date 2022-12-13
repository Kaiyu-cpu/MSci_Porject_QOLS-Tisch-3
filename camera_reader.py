import cv2
import numpy as np
import time
#%%
#setting up the camera
#camera = cv2.VideoCapture(0) #need to check further what the input 0 of this function
                             # means, this current version works on local machine

def get_image(camera,n,want_pic=False,t_delay = 0.125):
    '''
    the function that returns n image tensors
    params:
    camera | opencv camera object
    n | number of images specified
    want_pic | whether you want the picture files to be stored; Default is no
    returns:
    data: arrays of images
    '''
    data = np.zeros((n,1080,1920)) # size of the image, will think about image compression later
    for i in range(n):
        start = time.time()
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if want_pic == True:
            cv2.imwrite('image{}.png'.format(i), gray)
        gray = np.array(gray)
        data[i,:,:] = gray
        while (time.time()-start) < t_delay:
            continue
    return data


#%%
camera = cv2.VideoCapture(1)
start = time.time()
x = get_image(camera,16,want_pic = True)
end = time.time()
print(end-start)
