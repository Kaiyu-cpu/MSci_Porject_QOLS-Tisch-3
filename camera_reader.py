import cv2
import numpy as np
import time

#setting up the camera
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#%%
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
    data = np.zeros((n,64,64)) # size of the image, will think about image compression later
    for i in range(n):
        start = time.time()
        ret, frame = camera.read()
        print(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.array(gray)
        gray = cv2.resize(gray,(64,64))
        if want_pic == True:
            cv2.imwrite('image{}.png'.format(i), gray)
        data[i,:,:] = gray
        while (time.time()-start) < t_delay:
            continue
    return data

#%%

camera = cv2.VideoCapture(1)
start = time.time()
x = get_image(cap,2,want_pic = True)
end = time.time()
print(end-start)
