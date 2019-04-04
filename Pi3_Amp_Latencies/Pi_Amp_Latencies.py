#import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 

trials = 250;
blocks = 1;

#### Now we will go through the EEG file and determine our latencies
#filepath = ['M:\Data\GoPro_Visor\Pi_Amp_Latency_Test'];
#filename = ['testing_visor_pi_012.vhdr'];
#
#[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
#EEG = pop_loadbv(filepath, filename, [], []);
#[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'setname','timingtest','gui','off');
#
#
#start_trigger = ALLEEG(1).event(2).latency;
#EEG_latencies = zeros(1,trials);
#EEG_latencies = [];
#EEG_latency = 0;
#county = 1;


# dealing with scientific notation with numpy - np.format_float_scientific(np.float32(1.23e24), unique=False, precision=8)
# DataFrame.dropna get rid of na values
pi_recorded_times = pd.read_csv(r'C:\Users\User\Documents\GitHub\GoPro_Visor_Pi\Pi3_Amp_Latencies\Pi_Time_Data\012_visual'
                 '_p3_gopro_visor.csv', sep=',', header=None, dtype=np.float64)
pi_recorded_times = pi_recorded_times.T # transpose for plotting purposes
pi_recorded_times.columns = ['pi_type','pi_onset_latency','pi_delay','pi_resp','pi_jitter','pi_resp_latency','pi_start_stop'] # name the coloumns
#pi_recorded_times = pd.to_numeric(pi_recorded_times, errors='coerce')
pi_recorded_times = pi_recorded_times.apply(pd.to_numeric, args=('coerce',))  ## Convert to numeric
pi_recorded_times['pi_onset_latency'] = pi_recorded_times['pi_onset_latency'].astype('int')


# Column meanings
# trig type - 1 is standard 2 is target
# onset trig time latency
# delay length
# trial resp
# jitter length
# resp_latency
# start and stop of each block

# load both
# all_onset_latencies(1,:) = EEG_start_latencies;
all_onset_latencies = pi_recorded_times;
#
# conditions = {'EEG','Pi Times'};
pi_recorded_times = pi_recorded_times.T
# Plotting 
# Latency plot
sns.set()
fig = sns.relplot(x='pi_onset_latency', y='index', data=pi_recorded_times.reset_index());
fig.set(xlabel='Latency (Seconds)', ylabel='Trial Count')
plt.show(fig)


## Add extra row to have all lines start from 0:
#plot_table.loc['+0', :] = 0




ts = pd.Series(index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()

# Difference plot

tips = sns.load_dataset("tips")