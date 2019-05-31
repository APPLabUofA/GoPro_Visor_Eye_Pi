# -*- coding: utf-8 -*-

import numpy as np
import cv2
import pandas as pd

# Streamlined for the sake of quickly creating a video analysis script specifically for experiment 1

# Due to time constrains I am processing each video indivdually and am pulling/stitching together events from each after the fact
# Previous scripts can be used for picking out flash events in a variety of contexts

# %% Video Input Settings
for par in range(1,9): # 1,2,3,4,5,6,7,8
    if par != 2:       # 1,3,4,5,5,6,7,8
        for Vid_Num in range(1,3): # 1,2
            path = 'M:\\Data\\GoPro_Visor\\Experiment_1\\Video\\Converted\\Split\\00'
            in_format = '.avi'
            in_file = path + str(par) + '_0' + str(Vid_Num) + in_format # may need to add part in between the path and exp, depending on file name/exp_num
            
            # %% Video Output Settings - none
            
            
            # %% Experiment Specific Info
            trial_count = 750
            exp_num = 2
            broad_thresh = 1300000
            
            # %% Participant Specific Info
            
            if exp_num == 2 & Vid_Num == 2:                 
                start_flash = [0,0,0,0,0]
                past_last = [0,294960,0,486718,973436] #
                
            elif exp_num == 2:                               
                start_flash = [0,25000,0,15000,7500]
                past_last = [0,294960,0,215784,253950]
                
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
            
            col = ['green','blue','red']
            b_w8 = [0,0,0,1,1]
            g_w8 = [0,0,0,7,7]
            r_w8 = [0,0,0,1.4,1.4]
            
            
            # %% Main Analysis - Grabbing & Quanitfying video Frames
            cap = cv2.VideoCapture(in_file)
            cap.set(1, frame_number)
            fps = cap.get(cv2.CAP_PROP_FPS) 
            
            in_frame = True    
              
            while in_frame == True: 
                      
                if frame_number >= past_last[par]:
                    in_frame = False
                cap.set(1,frame_number)
                ret, frame = cap.read()  
                cv2.imshow("Base Image", frame)
                img1 = frame[240:480,212:636,:]
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
                        col_check = 4
                        ts_temp = frame_number
                # when change = 1 (during flash) & there is a last encounter with a flash train, then change 1 --> 0
                elif change == 1:
                    if temp_sum < broad_thresh:
                        change = 0
                        print("Flash Off")
                        
                # Moving window of averaging colour to increase accuracy of colour type detection
                if col_check > 0:
                    globals()['b' + str(col_check)], globals()['g' + str(col_check)], globals()['r' + str(col_check)] = cv2.split(img1)
                    if col_check == 1: 
                        b, g, r, =  (np.sum(b1) + np.sum(b2) + np.sum(b3) + np.sum(b4)), (np.sum(g1) + np.sum(g2) + np.sum(g3) + np.sum(g4)), (np.sum(r1) + np.sum(r2) + np.sum(r3) + np.sum(r4))
                        temp_max = 0,g*g_w8[par],b*b_w8[par],r*r_w8[par]
                        event_num = temp_max.index(max(temp_max))
                        Trigger_Start[ts_count] = ts_temp, event_num
                        print("Frame number {} is flash event {} is {} -  blue:{} green:{} red:{}".format(frame_number,ts_count+1,col[event_num-1],g*g_w8[par],b*b_w8[par],r*r_w8[par]))
                        ts_count += 1
                    col_check -= 1
            
                Trigger_State[count] = frame_number, change
            
                if change == 0:
                    last_frame = frame # make the current frame = to last_frame for drawing in the next iteration
            #    cv2.imshow('Original', frame)
            #    cv2.putText(img1, frame_number, (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA) 
                cv2.imshow('Decreased Size',img1)
                last_frame = img1
                if cv2.waitKey(10) & 0xFF == ord('q'):  # press q to quit
                    break
                count += 1
                frame_number += 1 # one by one
            # When everything done, release the capture
            cap.release() 
            cv2.destroyAllWindows()
            
            # %% Preprocessing
             
            # Construct a dataframe from the output of above scripts
            Trigger_Start_fin = Trigger_Start[0:ts_count,:]
            df1 = pd.DataFrame(Trigger_Start_fin) 
            df1.columns = ['Frame', 'Event'] # name columns - may need to add ['Adj_Index']
            df1 = df1.drop(df1[df1.Event ==3].index)
            df1 = df1.reset_index() #moves the index over - #df1 = df1.reset_index() # may need a second one to recalibrate index to index_0
            df1 = df1.drop(columns='index')
            df1['Frame'] = (df1['Frame'] - df1['Frame'][0]) #from each one minus the number of frames from the start of the first frame of the first red flash 
            df1['Frame'] = (df1['Frame']/fps) # Change from the conversion
            df1.columns = ['Time', 'Event']
            
            
            # Create temporary dataframes for each video
            if Vid_Num == 1:
                df1a = df1
                Leftover_Events = trial_count - len(df1a.index)
                export_csv = df1a.to_csv (r'M:\Data\GoPro_Visor\Experiment_1\Video_Times\Dataframe_df1a_00' + str(par) + '.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

            elif Vid_Num ==2: 
                # adjust based on offset of the first video - snip off the end based on expected events
                df1b = df1
                df1b['Time'] = df1b['Time'] + Vid_1_Dur[par]
#                df1b = df1b.reset_index()
#                df1b = df1b.drop(columns='index')
                df1b = df1b.drop(df1b.index[[list(range(Leftover_Events+1,len(df1b.index)))]]) #df1b.tail(1).index

                # Stitch together each video's temporary dataframes
                df1 = df1a.append(df1b, ignore_index=True) # concatenate event from seperate vides row-wise
                df1 = df1.drop(df1[df1.Event ==3].index)
                df1 = df1.reset_index()
                df1 = df1.drop(columns='index')
                #df1 = df1.drop(414)
                
                # Export only df1 to CSV - Or can also save a workspace 
                #export_csv = df1.to_csv (r'C:\Users\User\Desktop\export_dataframe_df1a_00' + str(par) + '.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
                export_csv = df1b.to_csv (r'M:\Data\GoPro_Visor\Experiment_1\Video_Times\Dataframe_df1b_00' + str(par) + '.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

                export_csv = df1.to_csv (r'M:\Data\GoPro_Visor\Experiment_1\Video_Times\Dataframe_df1_00' + str(par) + '.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
            
            

