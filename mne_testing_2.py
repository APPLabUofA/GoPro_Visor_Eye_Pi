# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:21:01 2019

@author: User
"""

import mne
import numpy as np
import matplotlib.pyplot as plt

#define some constants#
eeg_chan_num = 16
event_id = {'standards':1, 'targets':2}
colour = {1: 'black', 2: 'red'}
tmin, tmax = -0.2, 1.0
baseline = (None, 0.0)
reject1 = {'eeg': 1000, 'eog': 1000}
reject2 = {'eeg': 500, 'eog': 500}
plot_max_standards = 20
plot_min_standards = -10
plot_max_targets = 15
plot_min_targets = -5
electrodes = ['Pz']
colours = ['black', 'red']
conditions = ["Standards", "Targets"]

#load our data#
filename = 'M:\Data\GoPro_Visor\Pi_Amp_Latency_Test\testing_visor_pi_012.vhdr'
raw = mne.io.read_raw_brainvision(filename)

#specify montage, EOGs, and reference#
raw.set_montage('standard_1020')
raw.set_channel_types(mapping={'HEOG': 'eog'})
raw.set_channel_types(mapping={'VEOG': 'eog'})
mne.set_eeg_reference(raw.load_data(), ref_channels=['M2'])

#define our times, EEG and EOG channels#
eeg_data = raw[:eeg_chan_num,:][0]
eog_data = raw[eeg_chan_num:eeg_chan_num + 2,:][0]
times = raw[0,:][1]

#filter our raw data#
fmin, fmax = 0, 30 #in Hz
raw.filter(fmin, fmax)

#get our epochs#
picks = mne.pick_types(raw.info, eeg=True, eog=True)
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin,
                    tmax=tmax, baseline=baseline, reject=reject1, picks=picks)

#average our standards/targets together@
standards = epochs['standards'].average()
targets = epochs['targets'].average()

picks = [standards.ch_names.index(i) for i in electrodes], 
#now plot our ERPs for all channels#
#this will only plot electrode Pz with the axis flipped#
standards.plot(spatial_colors=True, gfp=True, 
                ylim=dict(eeg=[plot_max_standards,plot_min_standards]), xlim='tight', titles='Standard ERPs')
targets.plot(spatial_colors=True, gfp=True, 
                ylim=dict(eeg=[plot_max_targets,plot_min_targets]), xlim='tight', titles='Target ERPs')

#now let's compare our standard and target ERPs#
evoked_dict = dict(standards = epochs['standards'].average(), targets = epochs['targets'].average())

colors = dict(Left="Crimson", Right="CornFlowerBlue")
linestyles = dict(Auditory='-', visual='--')
pick = [evoked_dict["standards"].ch_names.index(i) for i in electrodes]

mne.viz.plot_compare_evokeds(evoked_dict, picks=pick, 
                             ylim=dict(eeg=[plot_max_targets,plot_min_targets]))

#difference waveform#
difference_waveform = epochs['standards'].average().data - epochs['targets'].average().data
ch_names = epochs['standards'].average().ch_names
plt.plot(difference_waveform[[ch_names.index(i) for i in electrodes],:])








#n_fft = 1024 # FFT size, should be a power of 2
#raw.notch_filter(np.arange(60, 241, 60), picks=picks, filter_length='auto',
#                 phase='zero')

#get our event information#
order = np.arange(raw.info['nchan'])
events = mne.find_events(raw)

mne.viz.plot_events(events, raw.info['sfreq'], raw.first_samp, color=colour,
                    event_id=event_id)
raw.plot(events=events, n_channels=18, order=order)

#perform a regression-based eye blink correction#
#right now this occurs on all the data#
#might want to epoch data and perform one round of artifact rejection first#
b = np.linalg.inv(eog_data @ eog_data.T) @ eog_data @ eeg_data.T

corrected_eeg_data_temp = (eeg_data.T - eog_data.T @ b).T
corrected_raw = raw.copy()
corrected_raw._data[:eeg_chan_num,:] = corrected_eeg_data_temp

raw.plot(n_channels=16, start=54, duration=60, 
              scalings=dict(eeg=50e-6))

corrected_raw.plot(n_channels=16, start=54, duration=60, 
              scalings=dict(eeg=50e-6))