# -*- coding: utf-8 -*-
# Note, must clear variable between reruns
import numpy as np
import cv2
import mne
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#import matplotlib.animation as animation

# %%
def equalizeHistColor(frame):
    # equalize the histogram of color image
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)  # convert to HSV
    img[:,:,2] = cv2.equalizeHist(img[:,:,2])     # equalize the histogram of the V channel
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)   # convert the HSV image back to RGB format

# %%
plt.ion()
colours = ['b','g','r']
col_name = ['','blue','green','red']
buff_size = 10

kernel = np.ones((5,5), np.uint8)
thresh = 10 # range that binary masks are derived from
thresh2 = 40 # take a more specific range of hsv values
sum_max_thresh = 40 # set this as the value 2 SD above baseline? - different for each channel? - seperate cahnnel weighting so max is reflective of baseline change
main_dict = {0:'frame_hist_',1:'norm_frame_hist_'}
col_dict = {'b_hsv':np.array((0,255,255),dtype=np.uint8),'g_hsv':np.array((60,255,255),dtype=np.uint8),'r_hsv':np.array((120,255,255),dtype=np.uint8),'b_bgr':[255,0,0],'g_bgr':[0,255,0],'r_bgr':[0,0,255]} #HSV values

d0={'frame_hist_current': np.zeros((3,256)),' test': np.zeros((3,256))}
d1={'norm_frame_hist_current': np.zeros((3,256)),' test': np.zeros((3,256))}


# %% Define Event Lists
Trigger_Start = np.zeros((1,2)) # List of [frame + start event trigger] (where the max[index] = corresponds with the last EEG event)
Trigger_Stop = np.zeros((1,2)) # List of [frame + end event trigger]
Trigger_State = np.zeros((1,2)) # List of [frame + trigger state] (0 B + G channels below thresholds, 1 above B channel threshold, 2 above R channel threshold
# Second Pass for extracting epochs based off first pass - figure out later
Trigger_Epoch = [] # Eventually will output an ~[-1,1] video epoch to be the raw input for deep learning
summy_summary = np.zeros((1,4))
last_frame = np.zeros((3,256))

hist_norm_b = hist_norm_g = hist_norm_r = 0

buffer_freeze = 0 # init buffer freeze state
param_quant = 0 # counts frames from two different methods
webcam = 0 # set to 1 if input is a webcam
exp_num = 2
broad_thresh = 200000
path = 'M:\\Data\\GoPro_Visor\\Experiment_1\\Video\\Converted\\Split\\00'
part = '003' # Version - example '001' or '054'
par = 3
Vid_Num = 1
anal = 0
in_format = '.avi'
in_file = path + str(par) + '_0' + str(Vid_Num) + in_format # may need to add part in between the path and exp, depending on file name/exp_num

if webcam == 1:
    in_file = 0

# for debugging purposes
#in_file = 'M:\\Data\\GoPro_Visor\\Experiment_1\\Video\\Converted\\003.avi'
#in_file = 'C:\\Users\\eredm\\OneDrive\\Desktop\\GOPR0212.avi'
# %% # Are we saving an output file (file with overlaid filters/bounders/manipulations)?
 # # Version - example '001' or '054'
out_format = '.avi'
out_file = path + str(par) + '_0' + str(Vid_Num) + '_output' + out_format
full_video = 0 # if == 0, will only output falshes, if == 1, will output every frame
# Output file parameter
#imgSize=(480,848) # likely best to set to original  dimensions
#frame_per_second=3
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#fourcc = cv2.VideoWriter_fourcc(*'XVID')


# %% Load in participant specific info - frame number of events
if exp_num == 1: # original exp (2018)
    start_eeg = [0,0,652,3041,3330,567,1045,2053,616,1443,638]
    door_closed = [0,0,4800,6947,12240,5040,7440,7680,6000,8640,8400]
    start_flash = [0,0,8897,12159,17668,10446,12567,12673,11040,13343,13176]
elif exp_num == 2 & Vid_Num == 2: # new data (2019) experiment 1, pilot 1
    start_flash = [0,0,0,0,0]
    past_last = [0,294960,0,486718,973436] #
elif exp_num == 2: # new data (2019) experiment 1, pilot 1
    start_flash = [0,25000,0,25000,12500]
    past_last = [0,294960,0,253950,973436] #294960

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
#    print(video_length + " " + frame_count)
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


# %% Start cycling through frames
cap = cv2.VideoCapture(in_file)
cap.get(3)
cap.get(4)
out = cv2.VideoWriter(out_file,fourcc, 20.0, (480,848))
## DEBUG WRITER
#frame = cv2.flip(frame,0)
#    # write the flipped frame
#    out.write(frame)
#    cv2.imshow('frame',frame)
#out = cv2.VideoWriter.open(out_file, fourcc, frame_per_second,imgSize,False)
count = 0

if webcam != 0:
        cap.set(1, start_flash[par])
skip_frames = 230 # baseline skip number of frames
frame_number = start_flash[par]
in_frame = True      
while in_frame == True: 
    change = 0      
    count += 1
    frame_number += skip_frames
    if frame_number >= past_last[par]:
        in_frame = False
    cap.set(1,frame_number)
    ret, frame = cap.read()  
    cv2.imshow(" ", frame)
    img1 = frame[240:480,212:636,:]

    if count == 1:
       last_frame = img1

    # Calculate if there was a change
    diff = cv2.absdiff(img1,last_frame)
    if np.sum(diff) >= broad_thresh: # replace this with a more colour specific filter (still quick)
        for i,col in enumerate(colours):
            globals()['hist_norm_' + str(col)] = cv2.calcHist([img1],[i],None,[256],[0,256])
            globals()['hist_norm_' + str(col)] = globals()['hist_norm_' + str(col)].reshape(1,-1)
            for i in range(3): # or just do it once and any single value over 1000 or something is set equal to zero, might be quicker
                globals()['hist_norm_' + str(col)][np.where(globals()['hist_norm_' + str(col)] == np.amax(globals()['hist_norm_' + str(col)]))] = 0    

        temp_sum = 0,np.sum(hist_norm_b[0,150:-1]),np.sum(hist_norm_g[0,200:-1]),np.sum(hist_norm_r[0,150:-1])
        change = temp_sum.index(max(temp_sum))

    Trigger_State = np.append(Trigger_State, np.matrix((frame_number, change)),axis=0)
      
    if change != 0:

        imgray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#        cv2.imshow('grey',imgray)
        dilation = cv2.dilate(imgray,kernel,iterations = 7)
        blur = cv2.GaussianBlur(dilation,(21,21),0) # take out if dilation is high?
#        cv2.imshow('dilation7',dilation)
        ret, thresh = cv2.threshold(blur, 20, 255, 0)
#        cv2.imshow('threshold',thresh)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) != 0:
            c = max(contours)
#            rect = cv2.boundingRect(c)
            height, width = img1.shape[:2]            
    #        if rect[2] > 0.2*height and rect[2] < 0.7*height and rect[3] > 0.2*width and rect[3] < 0.7*width: 
            x,y,w,h = cv2.boundingRect(c)            # get bounding box of largest contour
            img2 = cv2.drawContours(img1, c, -1, (255,255,255), 4)
            img3 = cv2.rectangle(img2,(x,y),(x+w,y+h),(0,0,255),2)  # draw red bounding box in img
            cv2.putText(img3, col_name[change], (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA) 
            cap.set(1,frame_number)
            ret, frame = cap.read()
            frame = frame[240:480,212:636,:]
            last_countie = frame[y:y+h,x:x+w,:]
#            cv2.imshow("",last_countie)
            cv2.imshow(' ',img3)  
            out.write(img3) # write the contour with label to a video
        else:
            if full_video == 1:
                out.write(img1)
        ######################################## Find Edges ######################
#            if Trigger_State[count+1] != Trigger_State[count+1]: #not using frame_number, but instead the last frame assesed = count + 1# only look in contoured frames that are not followed by another flash (as per the last Trigger_State)
            front_back = 1
            edge = []
            while front_back >= -1:
                previous_frame = img1
                framing_num = frame_number
                last_state_change = True
                direction = -1 * front_back # wondow direction - -1 = leftward, 1 = rightward
                i = 1
                detect = False
                jump_mod = 0 
                broad_threshold = 10000
                while detect == False:
                    jump = 100/(2*i)
    #                jump = 100^i
                    current_frame  = int(framing_num + jump*direction)
                    cap.set(1,current_frame)
                    ret, frame_current = cap.read() 
                    frame_current = frame_current[240:480,212:636,:]
                    countie = frame_current[y:y+h,x:x+w,:]
    #                cv2.imshow("last",last_countie)
    #                cv2.imshow("{}_frame_{}".format(i,current_frame),countie)
                    last_countie_sum = int(last_countie.sum())
                    countie_sum = int(countie.sum())
    #                diff = cv2.absdiff(frame_current, previous_frame)
                    if abs(last_countie_sum - countie_sum) >= broad_thresh:
                        direction = -1 * direction
                        state_change = True
                        i += 1
                    else:
                        state_change = False
                    if abs(framing_num - current_frame) < 2 and last_state_change == state_change:
    #                    if frame_number - current_frame > 0:
                            edge.append(current_frame) 
                            detect = True
                            print(i)
    #                    else:
    #                        i = 1
    #                        broad_threshold += 20000
                        ######################
    #                    for i in range(3):
    #                        currenty_frame = current_frame-i + i
    #                        cap.set(1, currenty_frame)
    #                        ret, frame_current = cap.read()
    #                        frame_current = frame_current[240:480,212:636,:]
    #                        print(currenty_frame)
    #                        cv2.imshow("frame_current_{}".format(i), frame_current)
    #                        cv2.putText(img3, col_name[change], (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA) 
    #                        out.write(frame_current)
                        ########################    
    
                    last_state_change = state_change
                    framing_num = current_frame
                    previous_frame = frame_current
                    last_countie = countie
    #                cv2.imshow("frame_current_{}".format(current_frame), countie)
                front_back -= 2
    #        findEdges()
            cap.set(1,frame_number)
    
            Trigger_Start = np.append(Trigger_Start, np.matrix((min(edge),change)), axis = 0)
            Trigger_Stop = np.append(Trigger_Stop, np.matrix((max(edge),change)), axis = 0)
            
# Display the original & resulting images
#    cv2.imshow('Change Detected', )
    last_count = count
    if change == 0:
        last_frame = frame # make the current frame = to last_frame for drawing in the next iteration
#    cv2.imshow('Original', frame)
#    cv2.imshow('Decreased Size',img1)
    last_frame = img1
    if cv2.waitKey(10) & 0xFF == ord('q'):  # press q to quit
        break
    
# When everything done, release the capture
out.release()
cap.release() 
cv2.destroyAllWindows()

# %% Confirm the efficacy of the flash pull script visually, for each flash (output is a video)

# if there are multiple identifications in a row that are not red, then take out the second event in the train
# look at the 2 frames around each start flash (3 in total) to ensure that we actually have the start of flashes, not the ends
#num_checks = 10
#cap = cv2.VideoCapture(in_file)
#for i in range(num_checks): #range(len(Trigger_Start))
#    currenty_frame = int(Trigger_Start[i+2,0])
#    for ii in range(3):
#        currentier_frame = currenty_frame-1 + ii
#        cap.set(1, currentier_frame)
#        ret, frame_current = cap.read()
#        frame_current = frame_current[240:480,212:636,:]
#        print(currentier_frame)
#        cv2.imshow("flash {} frame {}".format(i+1,ii-1), frame_current)
#        
#        
# %% Struture the data pulled from the video in a Pandas Data Frame

df1 = pd.DataFrame(Trigger_Start)
df1.columns = ['Frame', 'Event'] # name columns - may need to add ['Adj_Index']
df1 = df1.drop([0,0],axis=0) # df1.iloc[1:,] also works
df1 = df1.reset_index() #moves the index over - #df1 = df1.reset_index() # may need a second one to recalibrate index to index_0
df1 = df1.drop(columns='index')
df1['Frame'] = (df1['Frame'] - df1['Frame'][0]) #from each one minus the number of frames from the start of the first frame of the first red flash 
df1['Frame'] = (df1['Frame']/480) # Change from the conversion
df1.columns = ['Time', 'Event']
Targ_Std_diff = df1.diff()
Targ_Std_fin = df1.drop(Targ_Std_diff[Targ_Std_diff.Time < 0.2].index)
Targ_Std_fin = Targ_Std_fin.reset_index()
Targ_Std_fin = Targ_Std_fin.drop(columns='index')
Targ_Std_fin.columns = ['Time', 'Event']
Targ_Std_fin.insert(1, 'blank', 0) # needed for adding events
Targ_Std_fin.loc[Targ_Std_fin.Event == 1, 'Event'] = 22 # Targets
Targ_Std_fin.loc[Targ_Std_fin.Event == 2, 'Event'] = 21 # Standard
#Targ_Std_fin['Time'] = (Targ_Std_fin['Time'] * 1/240)
######################################### Load in EEG to compare

export_csv = Targ_Std_fin.to_csv (r'C:\Users\User\Desktop\export_dataframe.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

# %% Load in EEG times

filename = 'M:\Data\GoPro_Visor\Experiment_1\EEG_Data\\00' + str(par) + '_GoPro_Visor_Eye_Pi.vhdr' # pilot
raw = mne.io.read_raw_brainvision(filename, preload=True)
#raw.add_events = Targ_Std_fin.values
df2 = mne.find_events(raw) # outputs a numpy.ndarray
df2 = np.insert(df2,0,[0],axis = 0) #shift data one row down from the top so we don't miss the first event on o
df2 = pd.DataFrame(data=df2[1:,1:], index=df2[1:,0], columns=df2[0,1:])   # change to a pandas DataFrame
df2 = df2.reset_index() 
df2.columns = ['eeg_times', 'Empty', 'Event_Type'] # name columns
df2 = df2.drop(columns='Empty') # get rid of empty column
# align the MNE database event timings from the first target
df2['eeg_times'] = (df2['eeg_times'] - df2['eeg_times'][0]) * 0.001 # subtract all from start trigger - make sure it is the right trigger number

criteria_1 = df2['Event_Type'] == 1 
criteria_2 =  df2['Event_Type'] == 2
criteria_all = criteria_1 | criteria_2 # either/or event defined above
df2 = df2[criteria_all]
df2 = df2.reset_index() # resets index after removing events
df2 = df2.drop(columns='index')
## still need to minus all from the first event trigger before it gets deleted



# %% Create New Native EEG events - creating new epochs
#http://predictablynoisy.com/mne-python/auto_tutorials/plot_creating_data_structures.html#tut-creating-data-structures
#http://predictablynoisy.com/mne-python/auto_tutorials/plot_object_epochs.html
add_eeg_events = 'no'
if add_eeg_events == 'yes':
    mne.io.RawArray.add_events(raw,Targ_Std_fin.values, stim_channel=None, replace=False)
    stim_data = Targ_Std_fin.values
    info = mne.create_info(['STI'], raw.info['sfreq'], ['stim'])
    stim_raw = mne.io.RawArray(stim_data, info)
    raw.add_channels([stim_raw], force_update_info=True)



#%% Data Massage
df1 = Targ_Std_fin
df1 = df1.drop([0,0])
#df2 = df2.insert(df2,0,[0],axis = 0)
df2.loc[-1] = [2, 3]  # adding a row
df2.index = df2.index + 1  # shifting index
df2 = df2.sort_index()  # sorting by index

if Vid_Num == 1:
    df2a = df2
elif Vid_Num ==2:
    df2b = df2



#%% Plotting 
if anal ==1:
    df2 = np.concatenate(df2a,df2b) # concatenate event from seperate vides row-wise
    #all_onset_latencies = pd.concat([df1.assign(dataset='df1'), df2.assign(dataset='df2')])
    df3 = df1.join(df2) # join eeg_times to pi_times
    df3 = df3.reset_index()
    df3['Difference'] = df3['eeg_times'] - df3['Time']
    
    # Latency plot
    plt.close('all')
    plt.figure(0)
    plt.plot(df3['Time'], df3['index'], 'k--', label='Pi Times')
    plt.plot(df3['eeg_times'], df3['index'], 'ko', label='EEG Times')
    plt.xlabel('Latency (Seconds)')
    plt.ylabel('Trial Number')
    plt.title('Trial Number vs Latency -  Par_00{}'.format(par))
    legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')
    legend.get_frame().set_facecolor('C0')
    plt.show()
    
    # Difference plot
    plt.figure(1)
    plt.plot(df3['Difference'], df3['index'], label='EEG - Pi')
    legend = plt.legend(loc='upper left', shadow=True, fontsize='x-large')
    plt.xlabel('Latency (Seconds)')
    plt.ylabel('Trial Number')
    plt.title('Trial Number vs Difference - Par_00{}'.format(par))
    plt.show()
    
    
    # %% ##Linear Transform - do one for each block? Define by a ten second interval with no events
    trials = 398
    df4 = df3.copy() # copy DataFrame 
    df4 = df4.values # convert from Pandas DataFrame to a numpy structure
    #df4[:,12] = df4[:,1]-df4[:,5]
    df4 = np.append(df4, np.zeros((trials,3)), axis=1)
    
    
    # %% ## Transform the eeg_times to align with the Pi times - we then need to output each participant time as a single array
    ## loading it into each respective epoch dataframes as the updated times
    ## LinearRegression().fit(X, y) X=Training data (eeg_times), y=Target Values (pi_onset_latency)
    reg =  LinearRegression().fit(df4[:,1].reshape(-1,1), df4[:,5].reshape(-1,1))
    reg.score(df4[:,1].reshape(-1,1), df4[:,5].reshape(-1,1))
    df4[:,6] = reg.intercept_ + df4[:,1]*reg.coef_
    df4[:,7] = df4[:,5]-df4[:,6]
    # 1:eeg_times, 2:eeg_trig, 3:index, 4:pi times, 5:difference, 6:transformed difference, 7:difference between original difference and transformed difference


    # %% ## Transformed Difference plot
    plt.figure(2)
    plt.plot(df4[:,6], df4[:,0])
    #plt.plot(df4[:,10], df4[:,0]) #plot the magnitude of the difference 
    #plt.plot(df3['Difference'], df3['level_0'], label='EEG - Pi') # plot untransformed
    plt.legend('EEG - Pi', ncol=2, loc='upper left'); #  scalex=True, scaley=True if not using a custom xticks arguement
    plt.xlabel('Latency (miliseconds)')
    plt.ylabel('Trial Number')
    plt.title('Trial Number vs Transformed Difference')
    #plt.xlim([-0.001, 0, 0.001])
    plt.show()
    
    # %% Transform based on a linear regression based off of purely the first and last flashes
    # compare differences of each event
    
    #%% Comparison of Different Transforms
    Raw_Diff_Sum = sum(abs(df3['Difference']))
    





