# -*- coding: utf-8 -*-
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# %% Define Variables
par = 3

# %% Load in the Camera timing form the temporary workspace or cvs file for the participant you are working with
# If you want to work with group data - refer to Group_Figures.py
#df1 = pd.read_csv((r'C:\Users\User\Desktop\export_dataframe_df1c_00' + str(par) + '.csv', sep=',', header=True) # pilot
#df1 = np.insert(df2,0,[0],axis = 0) #shift data one row down from the top so we don't miss the first event on o
#df1 = pd.DataFrame(df1) 
#df1.columns = ['Frame', 'Event'] # name columns - may need to add ['Adj_Index']
#df1 = df1.reset_index() #moves the index over - #df1 = df1.reset_index() # may need a second one to recalibrate index to index_0
#df1 = df1.drop(columns='index')

# %% If you still need to stitch together videos use the following template to process

#Load old data if being used

#df1b = pd.read_csv((r'C:\Users\User\Desktop\export_dataframe_df1b_00' + str(par) + '.csv'), sep=',') # 
#df1b.columns = ['Frame', 'Event'] # name columns - may need to add ['Adj_Index']
#df1b = df1b.drop([0,0],axis=0) # df1.iloc[1:,] also works
#df1b = df1b.reset_index() #moves the index over - #df1 = df1.reset_index() # may need a second one to recalibrate index to index_0
#df1b = df1b.drop(columns='index')
#df1b.iloc[1:,]

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
df2['eeg_times'] = (df2['eeg_times'] - df2['eeg_times'][0]) * 0.001 # subtract all from start trigger (beginning of the first red flash) + convert to seconds

criteria_1 = df2['Event_Type'] == 1 
criteria_2 =  df2['Event_Type'] == 2
criteria_all = criteria_1 | criteria_2 # either/or event defined above
df2 = df2[criteria_all]
df2 = df2.reset_index() # resets index after removing events
df2 = df2.drop(columns='index')



# %% Plotting (Latency/Difference, Raw/Transformed(all point, 2-point, long-tailed), scatter/line/historgrams)
#all_onset_latencies = pd.concat([df1.assign(dataset='df1'), df2.assign(dataset='df2')])

## RAW CAMERA TIMES
df3 = df1.join(df2) # join eeg_times to pi_times
df3 = df3.reset_index()
df3['Difference (Seconds)'] = df3['eeg_times'] - df3['Time']
df3 = df3.dropna()
df3 = df3.drop(df3[abs(df3.Difference) > 2].index) # if there are still a few outliers - take them out with the following line
#df1 = df1.drop(df1[df1.Event ==3].index)
#Temp drop any rows from the df1 Dataframe and then rejoin with df2 to create a new df3
# this is not something we would normally have to do, working back from video only - we would just miss 1% of events -
# but can always make detection percentages better by tweaking detection parameters
#Targ_Std_diff = df1.diff()
#Targ_Std_fin = df1.drop(Targ_Std_diff[Targ_Std_diff.Time < 0.2].index)

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


#Seaborn Histogram Plots
#Event Latency
# EEG
df3['EEG Times (s)'] = df3['eeg_times']
plt.figure(figsize=(15,10))
plt.tight_layout()
plt.title('EEG Event Latency Distribution - Par_00{}'.format(par))
plt.ylabel('Number of Trials')
sns.distplot(df3['EEG Times (s)'])
# Camera
df3['Camera Times (s)'] = df3['Time']
plt.figure(figsize=(15,10))
plt.tight_layout()
plt.title('Camera Event Latency Distribution - Par_00{}'.format(par))
plt.ylabel('Number of Trials')
sns.distplot(df3['Camera Times (s)'])
# Difference
plt.figure(figsize=(15,10))
plt.tight_layout()
plt.title('EEG-Camera Difference Distribution - Par_00{}'.format(par))
plt.ylabel('Number of Trials')
sns.distplot(df3['Difference (s)'])


# %% ##Linear Transform 
trials = 441
df4 = df3.copy() # copy DataFrame 
df4 = df4.values # convert from Pandas DataFrame to a numpy structure
df4 = np.append(df4, np.zeros((trials,3)), axis=1)


# %% ## All Point Transform
## LinearRegression().fit(X, y) X=Training data (camera times), y=Target Values (eeg times)
model1 = LinearRegression()
X = df4[:,1].reshape(-1,1)
y = df4[:,3].reshape(-1,1)
reg =  model1.fit(X,y) # From the pi times we are predicting the eeg times
reg.score(df4[:,1].reshape(-1,1), df4[:,3].reshape(-1,1))
df4[:,6] = df4[:,1]*reg.coef_ + reg.intercept_  # eeg times = camera_times X slope of 
df4[:,7] = df4[:,3]-df4[:,6]
# 1:index , 2:pi times, 3:pi events, 1:eeg_times, 2:eeg_trig 5:difference, 6:transformed difference, 7:difference between original difference and transformed difference



# %% ## Transformed Difference plot
plt.figure(123)
plt.plot(df4[:,7], df4[:,0])
#plt.plot(df4[:,10], df4[:,0]) #plot the magnitude of the difference 
#plt.plot(df3['Difference'], df3['level_0'], label='EEG - Pi') # plot untransformed
plt.legend('EEG - Pi', ncol=2, loc='upper left'); #  scalex=True, scaley=True if not using a custom xticks arguement
plt.xlabel('Latency (miliseconds)')
plt.ylabel('Trial Number')
plt.title('Trial Number vs Transformed Difference {}'.format(par))
#plt.xlim([-0.00001, 0.00001])
plt.show()

# Difference
plt.figure(figsize=(15,10))
plt.tight_layout()
plt.title('EEG-Camera Difference Distribution - Par_00{}'.format(par))
plt.ylabel('Number of Trials')
plt.xlabel('Difference (seconds)')
sns.distplot(df4[:,7], rug = True, rug_kws={'color': 'black'})

# %% Transform based on a linear regression based off of purely the first and last flashes
# compare differences of each event

#%% Comparison of Different Transforms
Raw_Diff_Sum = sum(abs(df3['Difference']))


