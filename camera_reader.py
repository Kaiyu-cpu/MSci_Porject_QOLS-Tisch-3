import cv2
import numpy as np
import time


''' this cell below is the test for camera
while True:

    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
#%%
def Get_image(camera,want_pic=False):
    '''
    the function that returns an image array
    params:
    camera | opencv camera object
    n | number of images specified
    want_pic | whether you want the picture files to be stored
    returns:
    data: arrays of images
    '''
    
    data = np.zeros((256,256)) # size of the image, will think about image compression later
     
    
    
    ret, frame = camera.read()
        #print(frame)
    print("get image")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.array(gray)
    gray = cv2.resize(gray,(64,64))
    #if want_pic == True:
        #cv2.imwrite(f'iteration {iteration_num}, population {pop_num}', gray)
    data = gray
    
    return data

