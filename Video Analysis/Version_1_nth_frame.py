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
            
def Buffer_freezer(iii):
    for i in range(len(main_dict)):
        globals()['d' + str(i)][main_dict[i] + 'current'][iii,:] = hist

# %%
plt.ion()
colours = ['b','g','r']
col_name = ['blue','green','red']
buff_size = 10
thresh_weight = [] 
for i in range(buff_size):
    thresh_weight.append((1/2)**1/(i+1))
kernel = np.ones((5,5), np.uint8)
thresh = 10 # range that binary masks are derived from
thresh2 = 40 # take a more specific range of hsv values
sum_max_thresh = 50 # set this as the value 2 SD above baseline? - different for each channel? - seperate cahnnel weighting so max is reflective of baseline change
main_dict = {0:'frame_hist_',1:'norm_frame_hist_'}
col_dict = {'b_hsv':np.array((0,255,255),dtype=np.uint8),'g_hsv':np.array((60,255,255),dtype=np.uint8),'r_hsv':np.array((120,255,255),dtype=np.uint8),'b_bgr':[255,0,0],'g_bgr':[0,255,0],'r_bgr':[0,0,255]} #HSV values

d0={'frame_hist_current': np.zeros((3,256)),' test': np.zeros((3,256))}
d1={'norm_frame_hist_current': np.zeros((3,256)),' test': np.zeros((3,256))}

for i in range(buff_size):
     d0["frame_hist_" + str(i)] = np.zeros((3,256))
     
for i in range(buff_size):
     d1["norm_frame_hist_" + str(i)] = np.zeros((3,256))
     
for i in range(len(main_dict)):
    for iii, col in enumerate(colours):
        globals()['d' + str(i)][str(colours[iii]) + "_summary"] = np.zeros((buff_size,3))

for i, col in enumerate(colours): # broad values
    col_dict[str(col) + '_low_thresh_bro'] = np.array([col_dict[str(col) + '_hsv'][0] - thresh, 50, 50])
    col_dict[str(col) + '_high_thresh_bro'] = np.array([col_dict[str(col) + '_hsv'][0] + thresh, 255, 255])
#
for i, col in enumerate(colours): # specific values
    col_dict[str(col) + '_low_thresh_spe'] = np.array((col_dict[str(col) + '_hsv'][0] - thresh2, col_dict[str(col) + '_hsv'][1] - thresh2*3, col_dict[str(col) + '_hsv'][2] - thresh2*3))
    col_dict[str(col) + '_high_thresh_spe'] = np.array((col_dict[str(col) + '_hsv'][0] + thresh2, col_dict[str(col) + '_hsv'][1] + thresh2, col_dict[str(col) + '_hsv'][2] + thresh2))

# %% Define Event Lists
Trigger_Start = [0] # List of [frame + start event trigger] (where the max[index] = corresponds with the last EEG event)
Trigger_Stop = [0] # List of [frame + end event trigger]
Trigger_State = np.zeros((1,2)) # List of [frame + trigger state] (0 B + G channels below thresholds, 1 above B channel threshold, 2 above R channel threshold
# Second Pass for extracting epochs based off first pass - figure out later
Trigger_Epoch = [] # Eventually will output an ~[-1,1] video epoch to be the raw input for deep learning
summy_summary = np.zeros((1,4))

buffer_freeze = 0 # init buffer freeze state
param_quant = 0 # counts frames from two different methods
webcam = 1 # set to 1 if input is a webcam
exp_num = 2
broad_thresh = 20
path = 'M:\\Data\\GoPro_Visor\\Experiment_1\\Pilot_1\\GoPro_Videos\\Converted\\'
part = '011' # Version - example '001' or '054'
par = 1
exp = 'GOPR0212' # ex. '003_camera_p3'
in_format = '.avi'
in_file = path + exp + in_format # may need to add part in between the path and exp, depending on file name/exp_num
last_frame = np.zeros((3,256))


if webcam == 1:
    in_file = 0

# for debugging purposes
in_file = 'M:\\Data\\GoPro_Visor\\Experiment_1\\Pilot_1\\GoPro_Videos\\Converted\GOPR0212.avi'

# %% # Are we saving an output file (file with overlaid filters/bounders/manipulations)?
 # # Version - example '001' or '054'
out_format = '.avi'
out_file = path + exp + 'output' + out_format
full_video = 0 # if == 0, will only output falshes, if == 1, will output every frame
# Output file parameter
imgSize=(848,480) # likely best to set to original  dimensions
frame_per_second=240.0
out = cv2.VideoWriter(out_format, cv2.VideoWriter_fourcc(*"MJPG"), frame_per_second,imgSize,False)

# %% Load in participant specific info - frame number of events
if exp_num == 1: # original exp (2018)
    start_eeg = [0,0,652,3041,3330,567,1045,2053,616,1443,638]
    door_closed = [0,0,4800,6947,12240,5040,7440,7680,6000,8640,8400]
    start_flash = [0,0,8897,12159,17668,10446,12567,12673,11040,13343,13176]
elif exp_num == 2: # new data (2019) experiment 1, pilot 1
    start_flash = [0,15100]

# %% Quantify the parameters of the video
    
if param_quant == 1 & webcam == 0:
    # This can be off due to video compression + opencv source code shows it is calculated based on fps, which can be slightly off
        # for par 11 = 253950
    cap = cv2.VideoCapture(in_file)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ## The folowing script is a more refined version
        # for par 11 = 253956
    frame_count = 0
    while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break
    
            frame_count += 1
    print(video_length + " " + frame_count)
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# %% Initialize Plots
plt.ion()
fig, (ax1,ax2) = plt.subplots(2, sharex=True)
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
title = ax1.set_title("My plot", fontsize='large')

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

# %% Start cycling through frames
cap = cv2.VideoCapture(in_file)
count = 0

if webcam != 0:
        cap.set(1, start_flash[par])
skip_frames = 120 # baseline skip number of frames
frame_number = start_flash[par]
        
while(True): 
    
    change = 0
    ax1.clear()
    ax2.clear()
    ret, frame = cap.read()        
    count += 1
    frame_number += skip_frames
    cap.set(1,frame_number)
    

    for iii,col in enumerate(colours):
        

        hist = np.array([cv2.calcHist([frame],[iii],None,[256],[0,256])])
        hist = np.squeeze(hist, axis=(2))
        
        img1 = equalizeHistColor(frame)
        hsv_im = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

        hist_norm = cv2.calcHist([img1],[iii],None,[256],[0,256])
        hist_norm = hist_norm.reshape(1,-1)
        
        if buffer_freeze == 0 :
            if count > buff_size: 
                Buffer_update(iii)
            else:   
                Buffer_init(iii)
        elif buffer_freeze == 1:
            Buffer_freezer(iii)
        
        # This doesn't quite work yet, refer to 'hist_vid_test - Copy.py'
        ax1.plot(hist,color = col)
        ax2.plot(hist_norm, color = col)
        
    plt.draw()
    # Calculate if there was a change
    if count > buff_size: # implement a base cv2.absdiff filter that will speed up detection of, then goes back for x frames and recaptures buffer?
        if np.sum(cv2.absdiff(frame,last_frame)) >= broad_thresh: # replace this with a more colour specific filter (still quick)
             #weighted gradient of the past 10 frames in the buffer
#            subtrahend = np.zeros((3,256))
#            for ii in range(buff_size):
#                temp = thresh_weight[ii]*globals()['d' + str(i)][main_dict[i] + str(ii)]
#                subtrahend = np.add(subtrahend,temp)
#            weighted_subtrahend = sum(subtrahend)
            # the current frame minus the last frame, for both dicts
            difference = globals()['d1'][main_dict[1] + 'current']-globals()['d1'][main_dict[1] + '0']
            for iii,col in enumerate(colours):
                temp = difference[iii,100:-1]
                # calculate summary statistics from the high intensity pixels only
                diff_sum = average, std, sum = np.average(difference[iii,100:-1]), np.std(difference[iii,100:-1]), np.sum(difference[iii,100:-1])
                # add sum stats to appropriate dict
                # consider a weighted sum of pixels, so that the highest intensities are worth more
                globals()['d1'][str(colours[iii]) + "_summary"] = np.append(globals()['d1'][str(colours[iii]) + "_summary"],np.array([[int(average)],[int(std)],[int(sum)]]).T, axis = 0)
            # determine the event by comparing channel summary stats, particularily sum
            # remember we are dealing with the sum of difference in pixel in buckets from 100::, between frame x and frame x-1
            sum_summary = [d1['b_summary'][count-1,2], d1['g_summary'][count-1,2], d1['r_summary'][count-1,2],count]
            summy_summary = np.append(summy_summary,[sum_summary],axis=0)
            if max(sum_summary[0:3]) > sum_max_thresh: # doesn't include count
                change = sum_summary.index(max(sum_summary[0:3]))
                buffer_freeze = 1
        else:
            for iii, col in enumerate(colours):
                globals()['d1'][str(colours[iii]) + "_summary"] = np.append(globals()['d1'][str(colours[iii]) + "_summary"],np.array([[0,0,0]]), axis = 0)
    Trigger_State = np.append(Trigger_State, np.matrix((count, change)),axis=0)
      
    if int(Trigger_State[count-1][0,1]) == 0 & int(Trigger_State[count-2][0,1]) != 0:
        Trigger_Start.append(count)
    elif int(Trigger_State[count-1][0,1]) != 0 & int(Trigger_State[count-2][0,1]) == 0:
        Trigger_Stop.append(count)
    
    if change != 0:
        # The following 3 lines are for debugging - aren't needed
#        maskHSV = cv2.inRange(hsv_im, col_dict[str(col) + '_low_thresh_spe'], col_dict[str(col) + '_high_thresh_spe'])
#        resultHSV = cv2.bitwise_and(hsv_im, hsv_im, mask = maskHSV)
#        cv2.imshow("Result HSV", resultHSV)

#        img2 = cv2.addWeighted(frame,0.9,resultHSV,0.1,0)
#        cv2.imshow('addweight',img2)
        imgray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#        cv2.imshow('grey',imgray)
        dilation = cv2.dilate(imgray,kernel,iterations = 7)
        blur = cv2.GaussianBlur(dilation,(21,21),0) # take out if dilation is high?
#        cv2.imshow('dilation7',dilation)
        ret, thresh = cv2.threshold(blur, 100, 255, 0)
#        cv2.imshow('threshold',thresh)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) != 0:
            c = max(contours)
            rect = cv2.boundingRect(c)
            height, width = img1.shape[:2]            
    #        if rect[2] > 0.2*height and rect[2] < 0.7*height and rect[3] > 0.2*width and rect[3] < 0.7*width: 
            x,y,w,h = cv2.boundingRect(c)            # get bounding box of largest contour
            img4 = cv2.drawContours(img1, c, -1, (255,255,255), 2)
            img5 = cv2.rectangle(img1,(x,y),(x+w,y+h),(0,0,255),2)  # draw red bounding box in img
    
            cv2.putText(img5, col_name[change], (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA) 
            cv2.imshow('',img5)  
            out.write(img5)
        else:
            if full_video == 1:
                out.write(img1)
# Display the original & resulting images
#    cv2.imshow('Change Detected', )
    last_frame = hsv_im # make the current frame = to last_frame for drawing in the next iteration
    cv2.imshow('Original', frame)
    cv2.imshow('Histogram Equalization',img1)
    last_frame = frame
    if cv2.waitKey(10) & 0xFF == ord('q'):  # press q to quit
        break
        
# When everything done, release the capture
out.release()
cap.release()
cv2.destroyAllWindows()

# %%