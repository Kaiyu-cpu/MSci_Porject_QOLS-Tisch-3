import cv2
import numpy as np
from scipy.fft import fft, ifft

#setting up the camera
camera = cv2.VideoCapture(0) #need to check further what the input 0 of this function
                             # means, this current version works on local machine

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
    data = np.zeros((n,480,640)) # size of the image, will think about image compression later
    for i in range(n):
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if want_pic == True:
            cv2.imwrite('image{}.png'.format(i), frame)
            cv2.imwrite('image_gray{}.png'.format(i), gray)
        gray = np.array(gray)
        data[i,:,:] = gray
    return data


def Fourier_Transform(image):
    fft_image = fft(image)
    return fft_image