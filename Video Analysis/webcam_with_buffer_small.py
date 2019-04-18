# -*- coding: utf-8 -*-
# Note, must clear variable between reruns
import numpy as np
import cv2
from matplotlib import pyplot as plt
#import matplotlib.animation as animation

# %%
def equalizeHistColor(frame):
    # equalize the histogram of color image
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)  # convert to HSV
    img[:,:,2] = cv2.equalizeHist(img[:,:,2])     # equalize the histogram of the V channel
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)   # convert the HSV image back to RGB format

def Buffer_init(iii):
    for i in range(len(main_dict)):
        if count == 1:
            globals()['d' + str(i)][main_dict[i] + 'current'][iii,:] = hist
        elif count == 2:
            globals()['d' + str(i)][main_dict[i] + str(0)][iii,:] = globals()['d' + str(i)][main_dict[i] + 'current'][iii,:]
            globals()['d' + str(i)][main_dict[i] + 'current'][iii,:] = hist
        elif count > 2:
            for ii in range(count-1):
                print(ii)
                x = range(count)
                if ii < count - 1: #if count is equal to 3 (- 2) then i can only be 0, 0<1 == True
                    globals()['d' + str(i)][main_dict[i] + str(x[(ii*-1)-1])][iii,:] = globals()['d' + str(i)][main_dict[i] + str(x[(ii*-1)-2])][iii,:]
                else:               # if count is equal to 3, then we only want 1 iteration (ii*-1) - 1 = last = 2, (ii*-1) - 1 = 2nd to last = 1
                    pass
            globals()['d' + str(i)][main_dict[i] + str(0)][iii,:] = globals()['d' + str(i)][main_dict[i] + 'current'][iii,:]
            globals()['d' + str(i)][main_dict[i] + 'current'][iii,:] = hist
        if count == 10:
            b0 = d0 # baseline of hist
            b1 = d1 # baseline of norm_hist
            return b0, b1

def Buffer_update(iii):
    for i in range(len(main_dict)):
            for ii in range(buff_size-1):
                print(ii)
                x = range(buff_size)
                if ii < count - 1: 
                    globals()['d' + str(i)][main_dict[i] + str(x[(ii*-1)-1])][iii,:] = globals()['d' + str(i)][main_dict[i] + str(x[(ii*-1)-2])][iii,:]
                else:              
                    pass
            globals()['d' + str(i)][main_dict[i] + str(0)][iii,:] = globals()['d' + str(i)][main_dict[i] + 'current'][iii,:]
            globals()['d' + str(i)][main_dict[i] + 'current'][iii,:] = hist

# %%
plt.ion()
colours = ['b','g','r']

main_dict = {0:'frame_hist_',1:'norm_frame_hist_'}
buff_size = 10
col_dict = {b:[0,255,255],g:[60,255,255],r:[120,255,255]}
d0={'frame_hist_current': np.zeros((3,256)),' test': np.zeros((3,256))}
d1={'norm_frame_hist_current': np.zeros((3,256)),' test': np.zeros((3,256))}
for i in range(buff_size):
     d0["frame_hist_" + str(i)] = np.zeros((3,256))
for i in range(buff_size):
     d1["norm_frame_hist_" + str(i)] = np.zeros((3,256))
for i in range(len(main_dict)):
    for iii, col in enumerate(colours):
        globals()['d' + str(i)][str(colours[iii]) + "_summary"] = np.zeros((buff_size,3))
count = 0


for i, col in enumerate(colours)
    col_dict[globals()[str(colours[i]) + '_threshold' = col_dict[globals()[str(colours[i])]]
    thresh = 20
    minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
    maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])  
    maskBGR = cv2.inRange(bright,minBGR,maxBGR)
    resultBGR = cv2.bitwise_and(bright, bright, mask = maskBGR)
# %% Define Event Lists
Trigger_Start = [0] # List of [frame + start event trigger] (where the max[index] = corresponds with the last EEG event)
Trigger_Stop = [0] # List of [frame + end event trigger]
Trigger_State = np.zeros((1,2)) # List of [frame + trigger state] (0 B + G channels below thresholds, 1 above B channel threshold, 2 above R channel threshold
# Second Pass for extracting epochs based off first pass - figure out later
Trigger_Epoch = [] # Eventually will output an ~[-1,1] video epoch to be the raw input for deep learning
summy_summary = np.zeros((1,3))
# %% Initialize Plots
cap = cv2.VideoCapture(0)
plt.ion()
fig, (ax1,ax2) = plt.subplots(2, sharex=True)
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
title = ax1.set_title("My plot", fontsize='large')
sum_max_thresh = 100
# %% Start cycling through frames
while(True): 
    change = 0
    ax1.clear()
    ax2.clear()
    ret, frame = cap.read()        
    count += 1
    for iii,col in enumerate(colours):

        hist = np.array([cv2.calcHist([frame],[iii],None,[256],[0,256])])
        hist = np.squeeze(hist, axis=(2))
        print('debug')
        img = equalizeHistColor(frame)
        img1 = img

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
    # Calculate if there was a change
    if count > buff_size:
        for i in range(len(main_dict)):
             # the current frame minus the last frame, for both dicts
            difference = globals()['d' + str(i)][main_dict[i] + 'current']-globals()['d' + str(i)][main_dict[i] + '0']
            for iii,col in enumerate(colours):
                temp = difference[iii,100:-1]
                # calculate summary statistics from the high intensity pixels only
                diff_sum = average, std, sum = np.average(difference[iii,100:-1]), np.std(difference[iii,100:-1]), np.sum(difference[iii,100:-1])
                # add sum stats to appropriate dict
                # consider a weighted sum of pixels, so that the highest intensities are worth more
                globals()['d' + str(i)][str(colours[iii]) + "_summary"] = np.append(globals()['d' + str(i)][str(colours[iii]) + "_summary"],np.array([[int(average)],[int(std)],[int(sum)]]).T, axis = 0)
            # determine the event by comparing channel summary stats, particularily sum
            # remember we are dealing with the sum of difference in pixel in buckets from 100::, between frame x and frame x-1
        sum_summary = [d1['b_summary'][count-1,2], d1['g_summary'][count-1,2], d1['r_summary'][count-1,2]]
        summy_summary = np.append(summy_summary,[sum_summary],axis=0)
    if count > buff_size: 
        if max(sum_summary) > sum_max_thresh:
            change = sum_summary.index(max(sum_summary))
            # List of [frame + trigger state] (0 all channels below thresholds, 1 above B channel threshold, 2 above G channel threshold, above R channel threshold
            Trigger_State = np.append(Trigger_State, np.matrix((count, change)),axis=0)
        else:
            Trigger_State = np.append(Trigger_State, np.matrix((count,0)),axis=0)
    else:
        Trigger_State = np.append(Trigger_State, np.matrix((count,0)),axis=0)
    
    if count > 1:    
        if int(Trigger_State[count][0,1]) != 0 & int(Trigger_State[count-1][0,1]) == 0:
            Trigger_Start.append(count)
        elif int(Trigger_State[count][0,1]) != 0 & int(Trigger_State[count-1][0,1]) == 0:
            Trigger_Stop.append(count)
    
    if change != 0:
#        lower = [1, 0, 20]
#        upper = [60, 40, 200]
#        lower = np.array(lower, dtype="uint8")
#        upper = np.array(upper, dtype="uint8")
#        
#        mask = cv2.inRange(image, lower, upper)
#        output = cv2.bitwise_and(image, image, mask=mask)
#        ret,thresh = cv2.threshold(mask, 40, 255, 0)
#        im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # define range of blue color in HSV
        lower_blue = np.array([110,50,50])
        upper_blue = np.array([130,255,255])
    
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)
    
        cv2.imshow('frame',frame)
        cv2.imshow('mask',mask)
        cv2.imshow('res',res)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
            
        green = np.uint8([[[0,255,0 ]]])
        hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
        print('green hsv = {}'.format(hsv_green))
        [[[ 60 255 255]]]
        
        red = np.uint8([[[255,0,0 ]]])
        hsv_red = cv2.cvtColor(red,cv2.COLOR_BGR2HSV)
        print('red hsv = ' + hsv_red)
        
        blue = np.uint8([[[0,0,255]]])
        hsv_blue = cv2.cvtColor(blue,cv2.COLOR_BGR2HSV)
        print('blue hsv = ' + hsv_blue)
        
        
        
#        # red color boundaries (R,B and G)
#        lower = [1, 0, 20]
#        upper = [60, 40, 200]
#        
#        # create NumPy arrays from the boundaries
#        lower = np.array(lower, dtype="uint8")
#        upper = np.array(upper, dtype="uint8")
#        
#        # find the colors within the specified boundaries and apply
#        # the mask
#        mask = cv2.inRange(img1, lower, upper)
#        output = cv2.bitwise_and(img1, img1, mask=mask)
#        
#        ret,thresh = cv2.threshold(mask, 40, 255, 0)
#        contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#        
#        if len(contours) != 0:
#            # draw in blue the contours that were founded
#            cv2.drawContours(output, contours, -1, 255, 3)
#        
#            #find the biggest area
#            c = max(contours, key = cv2.contourArea)
#        
#            x,y,w,h = cv2.boundingRect(c)
#            # draw the book contour (in green)
#            cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
#        
#        cv2.imshow("Result", np.hstack([img1, output]))
#        
#        
        
        
        
#        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
#        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#        c = max(contours, key = cv2.contourArea)
#        x,y,w,h = cv2.boundingRect(c)
#        cv2.rectangle(imgray,(x,y),(x+w,y+h),(0,255,0),2)
##        cv2.drawContours(im, c, -1, (0,255,0), 3)
#        cv2.imshow('',imgray)
#        cv2.waitKey(10)

    
        
#        if len(contours) != 0:
#        for c in contours:
#            rect = cv2.boundingRect(c)
#            height, width = img2.shape[:2]            
#            if rect[2] > 0.2*height and rect[2] < 0.7*height and rect[3] > 0.2*width and rect[3] < 0.7*width: 
#                x,y,w,h = cv2.boundingRect(c)            # get bounding box of largest contour
#                img4=cv2.drawContours(img2, c, -1, color, thickness)
#                img5 = cv2.rectangle(img2,(x,y),(x+w,y+h),(0,0,255),2)  # draw red bounding box in img
#            else:
#                img5=img2
#    else:
#        img5=img2
        
# Display the original & resulting image
#    cv2.imshow('binary ', )
#    cv2.imshow('', )
#    cv2.imshow('', )
#    cv2.imshow('Change Detected', )
    cv2.imshow('Original', frame)
    cv2.imshow('Histogram Equalization',img)
    if cv2.waitKey(200) & 0xFF == ord('q'):  # press q to quit
        break
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()