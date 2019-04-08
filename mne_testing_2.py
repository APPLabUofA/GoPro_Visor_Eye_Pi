# -*- coding: utf-8 -*-

import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_morlet

#define some constants#
eeg_chan_num = 16
event_id = {'standards':1, 'targets':2}
colour = {1: 'black', 2: 'red'}
tmin, tmax = -0.2, 1.0
baseline = (None, 0.0)
reject1 = {'eeg': 1000, 'eog': 1000}
reject2 = {'eeg': 500, 'eog': 500}
plot_max_standards = 20 
plot_min_standards = -20#-10 
plot_max_targets = 20
plot_min_targets = -20#-5
electrodes = ['Pz']
colours = ['black', 'red']
conditions = ["Standards", "Targets"]

#load our data#
filename = 'M:\Data\GoPro_Visor\Pi_Amp_Latency_Test\\testing_visor_pi_011.vhdr'
filename = 'M:\Data\GoPro_P3_Latency\EEG_Data\\005_camera_p3.vhdr'
raw = mne.io.read_raw_brainvision(filename)

#specify montage, EOGs, and reference#
raw.set_montage('standard_1020')
raw.set_channel_types(mapping={'HEOG': 'eog'}) # set EOGs as such
raw.set_channel_types(mapping={'VEOG': 'eog'})
mne.set_eeg_reference(raw.load_data(), ref_channels=['M2']) # reference to mastoid

#define our times, EEG and EOG channels#
eeg_data = raw[:eeg_chan_num,:][0]
eog_data = raw[eeg_chan_num:eeg_chan_num + 2,:][0]
times = raw[0,:][1] # all times in 1 millisecond increments

#filter our raw data#
fmin, fmax = 0, 30 #in Hz
raw.filter(fmin, fmax) # high pass and low pass filters

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
#Topoplot
targets.plot_topomap(times=[0.4], size=3., title='Topoplot (Pz) @ 300 ms Post-stim', time_unit='s') #  titles='Topoplot'
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


target, standard = epochs["targets"].average, epochs["standards"].average
joint_kwargs = dict(ts_args=dict(time_unit='s'),
                    topomap_args=dict(time_unit='s'))
mne.combine_evoked([targets, standards], weights='equal').plot_joint(**joint_kwargs)

#n_fft = 1024 # FFT size, should be a power of 2
#raw.notch_filter(np.arange(60, 241, 60), picks=picks, filter_length='auto',
#                 phase='zero')

#get our event information#



########################################################### Start of PSD plots - still need to adapt
#frequencies =  np.linspace(6, 30, 100, endpoint=True)
#
#wave_cycles = 6
#
# # Compute morlet wavelet
#
## Left Cue
#tfr, itc = tfr_morlet(epochs['LeftCue'], freqs=frequencies, 
#                      n_cycles=wave_cycles, return_itc=True)
#tfr = tfr.apply_baseline([-1,-.5],mode='mean')
#tfr.plot(picks=[0], mode='logratio', 
#         title='TP9 - Ipsi');
#tfr.plot(picks=[1], mode='logratio', 
#         title='TP10 - Contra');
#power_Ipsi_TP9 = tfr.data[0,:,:]
#power_Contra_TP10 = tfr.data[1,:,:]
#
## Right Cue
#tfr, itc = tfr_morlet(epochs['RightCue'], freqs=frequencies, 
#                      n_cycles=wave_cycles, return_itc=True)
#tfr = tfr.apply_baseline([-1,-.5],mode='mean')
#tfr.plot(picks=[0], mode='logratio', 
#         title='TP9 - Contra');
#tfr.plot(picks=[1], mode='logratio', 
#         title='TP10 - Ipsi');
#power_Contra_TP9 = tfr.data[0,:,:]
#power_Ipsi_TP10 = tfr.data[1,:,:]
###############################################

order = np.arange(raw.info['nchan'])
events = mne.find_events(raw)

mne.viz.plot_events(events, raw.info['sfreq'], raw.first_samp, color=colour,
                    event_id=event_id)
raw.plot(events=events, n_channels=18, order=order)
#raw.plot_psd();
raw.plot_psd(fmin=1, fmax=30);
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





