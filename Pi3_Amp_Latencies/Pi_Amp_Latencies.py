#import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import mne

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
df1.columns = ['Latency_Time', 'Empty', 'Event_Type'] # name columns
df1 = df1.drop(columns='Empty') # get rid of empty column
df1['Latency_Time'] = (df1['Latency_Time'] - df1['Latency_Time'][0]) * 0.001 # subtract all from start trigger

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
df2 = df1[criteria_all]

# %%
# Combine the two into a single dataframe ? Nah, not for now
#all_onset_latencies = pd.concat([df1.assign(dataset='df1'), df2.assign(dataset='df2')])

#%%
# Plotting 
# Latency plot
sns.set()
df2 = sns.relplot(x='pi_onset_latency', y='index', data=pi_recorded_times.reset_index());
df1.set(xlabel='Latency (Seconds)', ylabel='Trial Count')
plt.show(fig)

# pandas.DataFrame.plot
df1 = df1.reset_index()
df1.plot(kind='line', x='index', y='Latency_Time')
plt.show()


ax = sns.lineplot(data = df1.reset_index(), x = 'index', y="Latency_Time")
plt.plot(,'Latency_Time', data=df1) # , marker='', color='olive', linewidth=2
plt.plot('index','pi_onset_latency', data=df2) # , marker='', color='olive', linewidth=2
# plt.set(xlabel='Latency (Seconds)', ylabel='Trial Count')
# plt.legend()
plt.show()




sns.relplot(x_vars=['Latency_Time'], y_vars=['Index'], data=df1, hue='Asset Subclass')
sns.relplot(x_vars=['pi_onset_latency'], y_vars=['Index'], data=df2, hue='Asset Subclass')
plt.show()



#import matplotlib
#matplotlib.__version__




## Add extra row to have all lines start from 0:
#plot_table.loc['+0', :] = 0


# Difference plot

tips = sns.load_dataset("tips")