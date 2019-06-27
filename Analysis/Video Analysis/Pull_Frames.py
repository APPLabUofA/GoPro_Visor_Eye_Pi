# -*- coding: utf-8 -*-
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

par = 4 # remember to change 
ws_name = {0:'Split',1:'Whole'}
whole_split = 1
Vid_Num = 1
first_flash = [0,0,0,15907,8512]

# Aligned Frame number of each event
Trigger_Frames_Aligned = Trigger_Start[0:ts_count,:]         # Pulls out all the events based on the number detected
df5 = pd.DataFrame(Trigger_Frames_Aligned)                   # Construct a Dataframe from the numpy array
df5.columns = ['Frame', 'Event']                        # Relabel coloumns
df5 = df5.drop(df5[df5.Event == 3].index)               # This will get rid of red events (start + end of blocks/experiment)
df5 = df5.reset_index()                                 # Moves the index over as a new coloumn
df5 = df5.drop(columns='index') 
df5 = df5.drop(df5.index[750:len(df5)])                     # 
#Leftover_Events = trial_count - len(df1.index)          # used to ensure the right number of events are found in the remaining video
df5a = df5
export_csv = df5a.to_csv (r'M:\Data\GoPro_Visor\Experiment_1\Video_Times\Dataframe_' + ws_name[whole_split] + '_Vid_0' + str(Vid_Num) + '_Par_00' + str(par) + '_Frames_Aligned.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path


#Trigger_Start[:,0] = Trigger_Start[:,0] - 15907
# Just Frame number of each event
Trigger_Frames = Trigger_Start[0:ts_count,:]         # Pulls out all the events based on the number detected
df4 = pd.DataFrame(Trigger_Frames)                  # Construct a Dataframe from the numpy array
df4.columns = ['Frame', 'Event']                        # Relabel coloumns
df4['Frame'] = df4['Frame'] + first_flash[par]
df4 = df4.drop(df4[df4.Event == 3].index)               # This will get rid of red events (start + end of blocks/experiment)
df4 = df4.reset_index()                                 # Moves the index over as a new coloumn
df4 = df4.drop(columns='index') 
df4 = df4.drop(df4.index[750:len(df4)])
df4a = df4
export_csv = df4a.to_csv (r'M:\Data\GoPro_Visor\Experiment_1\Video_Times\Dataframe_' + ws_name[whole_split] + '_Vid_0' + str(Vid_Num) + '_Par_00' + str(par) + '_Frames.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path


# %% Transform Frames Directly from Frames to times
# Load in EEG times

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
df2['eeg_times'] = (df2['eeg_times'] - df2['eeg_times'][0]) * 0.001 # subtract all from start trigger (beginning of the first red flash) + convert to seconds

criteria_1 = df2['Event_Type'] == 1 
criteria_2 =  df2['Event_Type'] == 2
criteria_all = criteria_1 | criteria_2 # either/or event defined above
df2 = df2[criteria_all]
df2 = df2.reset_index() # resets index after removing events
df2 = df2.drop(columns='index')

# Combined with time aligned frame numbers
df6 = df5a.join(df2) # join eeg_times to pi_times
df6 = df6.reset_index()


# Convert to numpy and transform
trials = 750
df7 = df6.copy() # copy DataFrame 
df7 = df7.values # convert from Pandas DataFrame to a numpy structure
df7 = np.append(df7, np.zeros((trials,6)), axis=1)
X1 = df7[:,1].reshape(-1,1)
y1 = df7[:,3].reshape(-1,1)
    #Equate and Test Regression
model1 = LinearRegression()
reg =  model1.fit(X1,y1) # From the pi times we are predicting the eeg times
reg.score(df7[:,1].reshape(-1,1), df7[:,3].reshape(-1,1))
df7[:,5] = df7[:,1]*reg.coef_ + reg.intercept_  # eeg times = camera_times X slope of + intercept
# Add new times to the dataframe 
df6['AP_Trans_Raw'] = df7[:,5] 
# Save
#df6 = df6.drop(columns='index')
df6a = df6
export_csv = df6a.to_csv (r'M:\Data\GoPro_Visor\Experiment_1\Video_Times\Dataframe_' + ws_name[whole_split] + '_Vid_0' + str(Vid_Num) + '_Par_00' + str(par) + '_Frames_Transformed.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path




