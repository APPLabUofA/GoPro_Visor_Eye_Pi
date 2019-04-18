# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
#import matplotlib.animation as animation

def equalizeHistColor(frame):
    # equalize the histogram of color image
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)  # convert to HSV
    img[:,:,2] = cv2.equalizeHist(img[:,:,2])     # equalize the histogram of the V channel
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)   # convert the HSV image back to RGB format
x = range(10)

def Buffer_init(iii):
    for i in range(2):
        if count == 1:
            globals()['d' + str(i)][main_dict[i] + 'current'][iii-1,256] = hist
        elif count == 2:
            globals()['d' + str(i)][main_dict[i] + str(1)][col-1,:] = globals()['d' + str(i)][main_dict[i] + 'current'][col-1,:]
            globals()['d' + str(i)][main_dict[i] + 'current'][col-1,:] = hist
        elif count > 2:
            for ii in range(count):
                x = range(count)
                if i < count - 2: #if count is equal to 3 (- 2) then i can only be 0, 0<1 == True
                    globals()['d' + str(i)][main_dict[i] + str(x[(ii*-1)-1])][col,:] = globals()['d' + str(i)][main_dict[i] + str(x[(ii*-1)-2])][col,:]
                else:               # if count is equal to 3, then we only want 1 iteration (ii*-1) - 1 = last, (ii*-1) - 1 = 2nd to last
                    pass
            globals()['d' + str(i)][main_dict[i] + str(1)][col,:] = globals()['d' + str(i)][main_dict[i] + 'current'][col,:]
            globals()['d' + str(i)][main_dict[i] + 'current'][col,:] = hist
        else:
            pass
        globals()['d' + str(i)][main_dict[i] + 'current'] = hist


def Buffer_update(col):
    for i in range(2):
            for ii in range(buff_size):
                x = range(10)
                if i < count - 2: #if count is equal to 3 (- 2) then i can only be 0, 0<1 == True
                    globals()['d' + str(i)][main_dict[i] + str(x[(ii*-1)-1])][col,:] = globals()['d' + str(i)][main_dict[i] + str(x[(ii*-1)-2])][col,:]
                else:               # if count is equal to 3, then we only want 1 iteration (ii*-1) - 1 = last, (ii*-1) - 1 = 2nd to last
                    pass
            globals()['d' + str(i)][main_dict[i] + str(1)][col,:] = globals()['d' + str(i)][main_dict[i] + 'current'][col,:]
            globals()['d' + str(i)][main_dict[i] + 'current'][col,:] = hist



plt.ion()
colours = ['b','g','r']

main_dict = {0:'frame_hist_',1:'norm_frame_hist_'}
buff_size = 10
d0={'frame_hist_current': np.zeros((3,256))}
d1={'norm_frame_hist_current': np.zeros((3,256))}
for i in range(buff_size):
     d0["frame_hist_" + str(i)] = np.zeros((3,256))
 
for i in range(buff_size):
     d1["norm_frame_hist_" + str(i)] = np.zeros((3,256))    
    
count = 0
# start video capture
cap = cv2.VideoCapture(0)
plt.ion()
fig, (ax1,ax2) = plt.subplots(2, sharex=True)
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
title = ax1.set_title("My plot", fontsize='large')

while(True):  # Capture frame-by-frame
    ax1.clear()
    ax2.clear()
    ret, frame = cap.read()        
    count += 1
    for iii,col in enumerate(colours):

        hist = np.array([cv2.calcHist([frame],[iii],None,[256],[0,256])])
        hist = np.squeeze(hist, axis=(2))
        print('debug')
        img = equalizeHistColor(frame)

        hist_norm = cv2.calcHist([img],[iii],None,[256],[0,256])
        hist_norm = hist_norm.reshape(1,-1)
        print('debug')
        
        if count > buff_size:
            Buffer_update(iii)
        else:   
            Buffer_init(iii)
        
        print('debug')

        ax1.plot(hist[100:256], color = col)
        ax2.plot(hist_norm[100:256], color = col)
    fig.canvas.draw()
    # Display the original & resulting image
    cv2.imshow('Original', frame)
    cv2.imshow('Histogram Equalization',img)
    if cv2.waitKey(100) & 0xFF == ord('q'):  # press q to quit
        break
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()