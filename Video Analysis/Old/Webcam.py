# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt

#plt.ion()
colours = ['b','g','r']
def equalizeHistColor(frame):
    # equalize the histogram of color image
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)  # convert to HSV
    img[:,:,2] = cv2.equalizeHist(img[:,:,2])     # equalize the histogram of the V channel
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)   # convert the HSV image back to RGB format

for i, col in enumerate(colours):
    globals()[str(colours[i]) + "_frame_hist"] = np.zeros((1,256))
    np.transpose(globals()[str(colours[i]) + "_frame_hist"])

for i, col in enumerate(colours):
	globals()[str(colours[i]) + "_frame_hist_values"] = np.zeros((1,3))

for i, col in enumerate(colours):
    globals()[str(colours[i]) + "_norm_frame_hist"] = np.zeros((1,256))
    np.transpose(globals()[str(colours[i]) + "_norm_frame_hist"])

for i, col in enumerate(colours):
    globals()[str(colours[i]) + "_norm_frame_hist_values"] = np.zeros((1,3))

count = 0
# start video capture
cap = cv2.VideoCapture(0)
fig, (ax1,ax2) = plt.subplots(2, sharex=True)
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
#fig.suptitle('Histograms')
ax1.set_xlabel('bins')
ax1.set_ylabel('number of bins')
ax1.set_title('Historgram')
ax2.set_title('Normalized Historgram')
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
        histr = np.array([cv2.calcHist([frame],[i],None,[256],[0,256])])
        histr = np.squeeze(histr, axis=(2))
#        hist_values = (np.mean(histr[100:-1]),np.std(histr[100:-1]),np.sum(histr[100:-1]))
#        globals()[str(colours[i]) + "_frame_hist_values"]= np.vstack(globals()[str(colours[i]) + "_frame_hist_values"],hist_values)
        globals()[str(colours[i]) + "_frame_hist"] = np.append(globals()[str(colours[i]) + "_frame_hist"],histr, axis=0)
        
        img = equalizeHistColor(frame)
        #####The same for the equalized histogram
        histr_norm = cv2.calcHist([img],[i],None,[256],[0,256]) 
#        histr_norm = np.array(histr)[np.newaxis]
#        np.transpose(histr_norm, (1,2,0))
#        histr_norm = np.squeeze(histr_norm, axis=(0,3))
#        np.reshape(histr_norm)
        histr_norm = histr_norm.reshape(1,-1)
        hist_norm_values = [np.mean(histr_norm[100:-1]),np.std(histr_norm[100:-1]),np.sum(histr_norm[100:-1])]
#        globals()[str(colours[i]) + "_norm_frame_hist_values"]= np.concatenate(globals()[str(colours[i]) + "_frame_hist_values"],hist_norm_values)
        globals()[str(colours[i]) + "_norm_frame_hist"] = np.append(globals()[str(colours[i]) + "_norm_frame_hist"],histr_norm, axis=0)
#        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)

        ax1.plot(histr,color = col)
        ax2.plot(histr_norm, color = col)
    plt.draw()
    # Display the resulting image
    cv2.imshow('Original', frame)
    cv2.imshow('Histogram Equalization',img)
    if cv2.waitKey(20) & 0xFF == ord('q'):  # press q to quit
        break
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()