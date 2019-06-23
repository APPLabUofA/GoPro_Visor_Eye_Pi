# -*- coding: utf-8 -*-

import numpy as np
import cv2
import pandas as pd

# TO RUN - ensure video is in the correct format (.avi) + video path + enter the particpant number and which of the 2 videos it is

# Streamlined for the sake of quickly creating a video analysis script specifically for experiment 1

# Due to time constrains I am processing each video indivdually and am pulling/stitching together events from each after the fact
# Previous scripts can be used for picking out flash events in a variety of contexts

# %% Video Input Settings
par = int(input('Which particpant video would you like to pull events from (single integer)? '))  # number 7 blue flashes are completely undetectable, number 8 has indestinguishable blue flashes (more red in them than blue)
in_format = '.avi'

whole_split = int(input('Would you like to analysis the whole video (1) or one of the the two segments (0)? ')) # 0 = split, 1 = whole
ws_name = {0:'Split',1:'Whole'}

if whole_split == 0:
    path = 'M:\\Data\\GoPro_Visor\\Experiment_1\\Video\\Converted\\Split\\00'
    Vid_Num = int(input('What video segement of participant {} would you like to process? '.format(par)))
    in_file = path + str(par) + '_0' + str(Vid_Num) + in_format # may need to add part in between the path and exp, depending on file name/exp_num
elif whole_split == 1:
    path = 'M:\\Data\\GoPro_Visor\\Experiment_1\\Video\\Converted\\Whole\\00'
    Vid_Num = 1
    in_file = path + str(par) + in_format # may need to add part in between the path and exp, depending on file name/exp_num

# %% Video Output Settings
out_format = '.avi'
out_file = path + str(par) + '_0' + str(Vid_Num) + '_output' + out_format
full_video = 0 # if == 0, will only output falshes, if == 1, will output every frame
# Output file parameter
#imgSize=(480,848) # likely best to set to original  dimensions
#frame_per_second=3
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#fourcc = cv2.VideoWriter_fourcc(*'XVID')


# %% Experiment Specific Info
trial_count = 750
exp_num = 2
broad_thresh = 1222000
stitch = 0 # combine both videos at the end of the script?

# %% Participant Specific Info
   
if Vid_Num == 1 & whole_split == 1: # new data (2019) experiment 1, pilot 1
    start_flash = [0,15000,0,15700,7500,18000,9500,14100,14000] #15000
    past_last = [0,294960,0,215784,253950,253950,256506,253950,253950] # 253950
  
if Vid_Num == 2 & whole_split == 0: # new data (2019) experiment 1, pilot 1
    start_flash = [0,0,0,0,0]
    past_last = [0,294960,0,486718,973436] #

if Vid_Num == 1 & whole_split == 1:
    start_flash = [0,15000,0,15700,7500,18000,9500,14100,14000] #15000
    past_last = [0,0,0,486718,452939,0,0]
 
total_frames = past_last[par]-start_flash[par]
Vid_1_Dur = [0,0,0,(253950-15000)/239.76023391812865,(253950-7500)/239.76023391812865] # The duration in time (s) of the first video (from the first flash - not including beginning)

# %% Initialize Data Containers
Trigger_State = np.zeros([total_frames,2]) # List of [frame + trigger state] (0 B + G channels below thresholds, 1 above B channel threshold, 2 above R channel threshold
Trigger_Start = np.zeros([trial_count*3,2])
Frame_Sum = np.zeros([total_frames,2])

# %% Initialize Variables
count = 0
frame_number = start_flash[par]
change = 0
col_check = 0
ts_count = 0
event_num = 0
kernel = np.ones((5,5), np.uint8)
event_num = 2


col = ['green','blue','red']
b_w8 = [0,1.2,0,1.5,1.5,1,1.6,1.3,1.4]
g_w8 = [0,7,0,6,7,7,9,8,7.5]
r_w8 = [0,1.2,0,1.55,1.3,1,1.35,1.3,1.4,1.05]


# %% Video Manipulation Functions

def equalizeHistColor(frame):
    # equalize the histogram of color image
#    img = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)  # convert to HSV
#    img[:,:,2] = cv2.equalizeHist(img[:,:,2])     # equalize the histogram of the V channel
#    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)   # convert the HSV image back to RGB format
    image_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    img1 = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
    return img1

# %% Main Analysis - Grabbing & Quanitfying video Frames
cap = cv2.VideoCapture(in_file)
out = cv2.VideoWriter(out_file,fourcc, 40, (424,240))
cap.set(1, frame_number)
fps = cap.get(cv2.CAP_PROP_FPS) 

in_frame = True    
  
while in_frame == True: 
          
    if frame_number >= past_last[par]-1:
        in_frame = False
    cap.set(1,frame_number)
    ret, frame = cap.read()  
#    cv2.imshow("Base Image", frame)
    img1 = frame[240:480,212:636,:]
#    img1 = equalizeHistColor(img1)
    temp_sum = np.sum(img1)
    Frame_Sum[count] = frame_number, temp_sum
    if count == 10:
        baseline = np.average(Frame_Sum[0:10,1])
    if count % 50 == 0 :
        print(temp_sum)
    # when change = 0 (during no flash) & there is a first encounter with a flash train, then change 0 --> 1  
    if change == 0:
        if temp_sum > broad_thresh:
            change = 1
            print("Flash On")
            col_check = 5
            ts_temp = frame_number
    # when change = 1 (during flash) & there is a last encounter with a flash train, then change 1 --> 0
    elif change == 1:
        if temp_sum < broad_thresh:
            change = 0
            print("Flash Off")
    if col_check > 0:
        imgray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#        cv2.imshow('grey',imgray)
        dilation = cv2.dilate(imgray,kernel,iterations = 7)
        blur = cv2.GaussianBlur(dilation,(21,21),0) # take out if dilation is high?
#        cv2.imshow('dilation7',dilation)
        ret, thresh = cv2.threshold(blur, 5, 255, 0)
#        cv2.imshow('threshold',thresh)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) != 0:
            c = max(contours, key = cv2.contourArea)
#            rect = cv2.boundingRect(c)
            height, width = img1.shape[:2]            
    #        if rect[2] > 0.2*height and rect[2] < 0.7*height and rect[3] > 0.2*width and rect[3] < 0.7*width: 
            x,y,w,h = cv2.boundingRect(c)            # get bounding box of largest contour
            img2 = cv2.drawContours(img1, c, -1, (255,255,255), 4)
            img3 = cv2.rectangle(img2,(x,y),(x+w,y+h),(0,0,255),2)  # draw red bounding box in img


        # Moving window of averaging colour to increase accuracy of colour type detection

        globals()['b' + str(col_check)], globals()['g' + str(col_check)], globals()['r' + str(col_check)] = cv2.split(img3)
        if col_check == 1: 
            b, g, r, =  (np.sum(b1) + np.sum(b2) + np.sum(b3) + np.sum(b4)), (np.sum(g1) + np.sum(g2) + np.sum(g3) + np.sum(g4)), (np.sum(r1) + np.sum(r2) + np.sum(r3) + np.sum(r4))
            temp_max = int(g*g_w8[par]),int(b*b_w8[par]),int(r*r_w8[par])
            event_num = temp_max.index(max(temp_max))
            Trigger_Start[ts_count] = ts_temp, event_num+1
            print("Frame number {} is flash event {} is {} -   green:{} -  blue:{} -  red:{}".format(frame_number,ts_count+1,col[event_num],int(g*g_w8[par]),int(b*b_w8[par]),int(r*r_w8[par])))
            ts_count += 1

        
            if len(contours) != 0:
                cv2.putText(img3, col[event_num], (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA) 
                cv2.putText(img3, str(frame_number), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA) 
        #            cap.set(1,frame_number)
        #            ret, frame = cap.read()
        #            frame = frame[240:480,212:636,:]
        #            last_countie = frame[y:y+h,x:x+w,:]
                cv2.imshow('finished',img3)
                out.write(img3) # write the contour with label to a video
        col_check -= 1
            
    Trigger_State[count] = frame_number, event_num

    if change == 0:
        last_frame = frame # make the current frame = to last_frame for drawing in the next iteration
#    cv2.imshow('Original', frame)
#    cv2.putText(img1, frame_number, (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA) 
    cv2.imshow('Every Frame',img1)
#    out.write(img1)
    last_frame = img1
    if cv2.waitKey(10) & 0xFF == ord('q'):  # press q to quit
        break
    count += 1
#    event_num = 0
    frame_number += 1 # one by one
# When everything done, release the capture
cap.release() 
out.release()
cv2.destroyAllWindows()

# %% Preprocessing
# Differs depending on whether it is Video 1 or 2

# Create temporary dataframes for each video
if Vid_Num == 1:
    Trigger_Start_fin = Trigger_Start[0:ts_count,:]         # Pulls out all the events based on the number detected
    df1 = pd.DataFrame(Trigger_Start_fin)                   # Construct a Dataframe from the numpy array
    df1.columns = ['Frame', 'Event']                        # Relabel coloumns
    df1['Frame'] = (df1['Frame'] - df1['Frame'][0])         # From each one minus the number of frames from the start of the first frame of the first red flash 
    df1 = df1.drop(df1[df1.Event == 3].index)               # This will get rid of red events (start + end of blocks/experiment)
    df1['Frame'] = (df1['Frame']/fps)                       # Change from frame number to seconds from the start of the experiment
    df1.columns = ['Time', 'Event']                         # Rename coloumn to reflect time 
    temp_diff = df1.diff()                                  # Take the difference vertically to find the time gap between each event
    df1 = df1.drop(temp_diff[(temp_diff.Time < 1.5)].index) # |((temp_diff.Time[2:-1] > 3)&(temp_diff.Time[2:1] < 7)) # This will get rid of double detections (events within 0.2 seconds of each other)
    df1 = df1.reset_index()                                 # Moves the index over as a new coloumn
    df1 = df1.drop(columns='index')                         # 
    Leftover_Events = trial_count - len(df1.index)          # used to ensure the right number of events are found in the remaining video
    df1a = df1
    export_csv = df1a.to_csv (r'M:\Data\GoPro_Visor\Experiment_1\Video_Times\Dataframe_' + ws_name[whole_split] + '_Vid_0' + str(Vid_Num) + '_Par_00' + str(par) + '.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

elif Vid_Num == 2:                              # Very similar except that we do not subtract all events from the first - here we also compare detected events against expected
    Trigger_Start_fin = Trigger_Start[0:ts_count,:]
    df1 = pd.DataFrame(Trigger_Start_fin)
    df1.columns = ['Frame', 'Event']
    df1 = df1.drop(df1[df1.Event == 3].index)   # This will get rid of red events (start + end of blocks/experiment)
    df1['Frame'] = (df1['Frame']/fps)           # Change from frame number to seconds from the start of the experiment
    df1.columns = ['Time', 'Event']             # Rename coloumn to reflect time 
    temp_diff = df1.diff()                      # take the difference vertically to find the time gap between each event
    df1 = df1.drop(temp_diff[(temp_diff.Time < 1.5)].index) 
    df1 = df1.reset_index()
    df1 = df1.drop(columns='index')
    df1b = df1
    df1b['Time'] = df1b['Time'] + Vid_1_Dur[par] # add the offset of the number of frames since the first flash (red) in ther first video
    df1b = df1b.reset_index()
    df1b = df1b.drop(columns='index')
    # Shouldn't have to use the next line, but it will cut down the events to the number of expected events given the number found in the first video - cuts form the end)
#    df1b = df1b.drop(df1b.index[[list(range(Leftover_Events+1,len(df1b.index)))]]) #df1b.tail(1).index
    export_csv = df1b.to_csv (r'M:\Data\GoPro_Visor\Experiment_1\Video_Times\Dataframe_' + ws_name[whole_split] + '_Vid_0' + Vid_Num + 'Par_00' + str(par) + '.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

if stitch == 1:
    # Stitch together each video's temporary dataframes
    df1 = df1a.append(df1b, ignore_index=True) # concatenate event from seperate vides row-wise
    df1 = df1.drop(df1[df1.Event ==3].index)
    df1 = df1.reset_index()
    df1 = df1.drop(columns='index')
    #df1 = df1.drop(414)

## Export only df1 to CSV - Or can also save a workspace manually
# export_csv = df1.to_csv (r'C:\Users\User\Desktop\export_dataframe_df1a_00' + str(par) + '.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

#    export_csv = df1.to_csv (r'M:\Data\GoPro_Visor\Experiment_1\Video_Times\Dataframe_' + ws_name[whole_split] + '_Vid_Full_Par_00' + str(par) + '.csv' index = None, header=True) #Don't forget to add '.csv' at the end of the path
            
            

