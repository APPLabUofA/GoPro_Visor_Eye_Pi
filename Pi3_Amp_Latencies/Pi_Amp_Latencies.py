#import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.linear_model import LinearRegression
import mne

plt.close('all')

trials = 250;
blocks = 1;

# # The only thing you need to change is going to be par (participant number) the rest will be dictated by dictionaries
par = "012"


# %% We will now load in the EEG data 
#### Now we will go through the EEG file and determine our latencies

filename = 'M:\Data\GoPro_Visor\Pi_Amp_Latency_Test\\testing_visor_pi_' + par + '.vhdr'
raw = mne.io.read_raw_brainvision(filename)
df1 = mne.find_events(raw) # outputs a numpy.ndarray
df1 = np.insert(df1,0,[0],axis = 0) #shift data one row down from the top so we don't miss the first event on o
df1 = pd.DataFrame(data=df1[1:,1:], index=df1[1:,0], columns=df1[0,1:])   # change to a pandas DataFrame
df1 = df1.reset_index() 
df1.columns = ['eeg_times', 'Empty', 'Event_Type'] # name columns
df1 = df1.drop(columns='Empty') # get rid of empty column
df1['eeg_times'] = (df1['eeg_times'] - df1['eeg_times'][0]) * 0.001 # subtract all from start trigger

criteria_1 = df1['Event_Type'] == 1 
criteria_2 =  df1['Event_Type'] == 2
criteria_all = criteria_1 | criteria_2 # either/or event defined above
df1 = df1[criteria_all]
df1 = df1.reset_index() # resets index after removing events
df1 = df1.drop(columns='index')
## still need to minus all from the first event trigger before it gets deleted


# %% Here we extract thhe Pi times (imported as df2)
df2 = pd.read_csv((r'C:\Users\User\Documents\GitHub\GoPro_Visor_Pi\Pi3_Amp_Latencies\Pi_Time_Data\012_visual_p3_gopro_visor.csv'), sep=',', header=None)
df2 = df2.T # transpose for plotting purposes
df2.columns = ['pi_type','pi_onset_latency','pi_delay','pi_resp','pi_jitter','pi_resp_latency','pi_start_stop'] # name the coloumns
df2 = df2.apply(pd.to_numeric, args=('coerce',))  ## Convert to numeric
df2= df2.dropna(thresh=2)
criteria_1 = df2['pi_type'] == 1 
criteria_2 =  df2['pi_type'] == 2
criteria_all = criteria_1 | criteria_2
df2 = df1[criteria_all] # Unalignable boolean Series provided as indexer (index of the boolean Series and of the indexed object do not match
# deal with this with the following - df2 = df2.reset_index()
# %% 
df2 = df2.reset_index()
# %%
# Combine the two into a single dataframe ? Nah, not for now
#all_onset_latencies = pd.concat([df1.assign(dataset='df1'), df2.assign(dataset='df2')])
df3 = df1.join(df2) # join eeg_times to pi_times
df3 = df3.reset_index()
df3['Difference'] = df3['eeg_times'] - df3['pi_onset_latency']
#%%
# Plotting 
# Latency plot
plt.close('all')
# matlibplot 
plt.plot(df3['pi_onset_latency'], df3['level_0'], 'k--', label='Pi Times')
plt.plot(df3['eeg_times'], df3['level_0'], 'ko', label='EEG Times')
# plt.legend('EP', ncol=2, loc='upper left'); # Figure legend
plt.xlabel('Latency (Seconds)')
# plt.ylabel('Trial Count')
legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')
# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C0')
plt.show()

# Difference plot
plt.plot(df3['Difference'], df3['level_0'])
plt.legend('EEG - Pi', ncol=2, loc='upper left'); # Figure legend
plt.xlabel('Latency (Seconds)')
# plt.ylabel('Trial Count')
plt.show()

# %% ##Linear Transform
df4 = df3.copy() # copy DataFrame 
df4 = df4.values # convert from Pandas DataFrame to a numpy structure

# %% ## Transform the eeg_times to align with the Pi times - we then need to output each participant time as a single array
## loading it into each respective epoch dataframes as the updated times
## LinearRegression().fit(X, y) X=Training data (eeg_times), y=Target Values (pi_onset_latency)
reg =  LinearRegression().fit(df4[:,1].reshape(-1,1), df4[:,5].reshape(-1,1))
reg.score(df4[:,1].reshape(-1,1), df4[:,5].reshape(-1,1))

# %% ## Transformed Difference plot
plt.plot(df4[:,11], df4[:,0])
plt.legend('EEG - Pi', ncol=2, loc='upper left'); #  scalex=True, scaley=True if not using a custom xticks arguement
plt.xlabel('Latency (Seconds)')
plt.xticks([-0.001, 0, 0.001])
plt.show()




