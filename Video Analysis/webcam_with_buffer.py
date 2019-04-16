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

def Buffer_init():
    for i in range(2):
        if count > 1:
            for ii in range(count):
                if ii == 0:
                    break
                else:
                    globals()['d' + str(i)][main_dict[i] + str((ii*-1) - 1)] = globals()['d' + str(i)][main_dict[i] + str((ii*-1) - 2)]
        else:
            pass
        globals()['d' + str(i)][main_dict[i] + 'current'] = hist

def Buffer_update():
    for i in main_dict:
            for ii in range(count-1):
                globals()['d' + str(i)][main_dict[i] + str(ii+1)] = globals()['d' + str(i)][main_dict[i] + str(ii)]
            globals()['d' + str(i)][main_dict[i] + 'current'] = hist     
         


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


#fig = plt.figure()
#ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
#                   xticklabels=[], ylim=(-1.2, 1.2))
#ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4],
#                   ylim=(-1.2, 1.2))
ax1.tick_params(axis='x',          # changes apply to the x-axis
       which='both',      # both major and minor ticks are affected
       bottom=False,      # ticks along the bottom edge are off
       top=False,         # ticks along the top edge are off
       labelbottom=False) # labels along the bottom edge are off)
#fig.suptitle('Histograms')
   
    #plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='on')
#fig.tick_params(axis='y', which='both', left='off' labelbottom='on')

while(True):
    # Capture frame-by-frame
    ax1.clear()
    ax2.clear()
    ret, frame = cap.read()        
    count += 1
    # Our operations on the frame come here    
    #img = frame

    for i,col in enumerate(colours):
#        histr = np.array([cv2.calcHist([frame],[i],None,[256],[0,256])])
#        histr = np.squeeze(histr, axis=(2))
#       globals()[str(colours[i]) + "_frame_hist"] = np.append(globals()[str(colours[i]) + "_frame_hist"],histr, axis=0)
#        img = equalizeHistColor(frame)
#        #####The same for the equalized histogram
#        histr_norm = cv2.calcHist([img],[i],None,[256],[0,256]) 
#      histr_norm = histr_norm.reshape(1,-1)
#        hist_norm_values = [np.mean(histr_norm[100:-1]),np.std(histr_norm[100:-1]),np.sum(histr_norm[100:-1])]
##        globals()[str(colours[i]) + "_norm_frame_hist_values"]= np.concatenate(globals()[str(colours[i]) + "_frame_hist_values"],hist_norm_values)
#        globals()[str(colours[i]) + "_norm_frame_hist"] = np.append(globals()[str(colours[i]) + "_norm_frame_hist"],histr_norm, axis=0)
##        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
#      
        hist = np.array([cv2.calcHist([frame],[i],None,[256],[0,256])])
        hist = np.squeeze(hist, axis=(2))
        print('debug')
        img = equalizeHistColor(frame)

        hist_norm = cv2.calcHist([img],[i],None,[256],[0,256])
        hist_norm = hist_norm.reshape(1,-1)
        print('debug')
        
        if count > buff_size:
            Buffer_update()
        else:   
            Buffer_init()
        
        print('debug')

        ax1.plot(hist[100:256], color = col)
        ax2.plot(hist_norm[100:256], color = col)
    fig.canvas.draw()
    # Display the resulting image
    cv2.imshow('Original', frame)
    cv2.imshow('Histogram Equalization',img)
    if cv2.waitKey(100) & 0xFF == ord('q'):  # press q to quit
        break
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()