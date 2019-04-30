# -*- coding: utf-8 -*-

import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_morlet, psd_multitaper
from sklearn.cluster.spectral import spectral_embedding  # noqa
from sklearn.metrics.pairwise import rbf_kernel   # noqa

#define some constants#
eeg_chan_num = 16
event_id = {'standards':1, 'targets':2}
colour = {1: 'black', 2: 'red'}
tmin, tmax = -0.2, 1.0
baseline = (None, 0.0)
reject1 = {'eeg': 1000, 'eog': 1000}
reject2 = {'eeg': 500, 'eog': 500}
plot_max_standards = 25 
plot_min_standards = -25#-10 
plot_max_targets = 25
plot_min_targets = -25#-5
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

#average our standards/targets together + 
standards = epochs['standards'].average()
targets = epochs['targets'].average()

# %% ##ERPs for all channels#
standards.plot(spatial_colors=True, gfp=True, 
                ylim=dict(eeg=[plot_max_standards,plot_min_standards]), xlim='tight', titles='Standard ERPs')
targets.plot(spatial_colors=True, gfp=True, 
                ylim=dict(eeg=[plot_max_targets,plot_min_targets]), xlim='tight', titles='Target ERPs')

# %% ##Topoplot##

mne.viz.plot_evoked_topomap(targets, times=[0.3, 0.35, 0.4], average=0.05, size=3., cmap=matplotlib colormap, title='Topoplot (Pz) @ 300 ms Post-stim - Targets', time_unit='s', vmin=-15, vmax=15)
mne.viz.plot_evoked_topomap(standards, times=[0.35], size=3., average=0.05, title='Topoplot (Pz) @ 300 ms Post-stim - Standards', time_unit='s', vmin=-15, vmax=15) 
# averages are evenly split around the time window - so average=0.01 averages over 0.005<x>0.005

targets.animate_topomap(ch_type='eeg', times=np.arange(0.0,0.6,0.02), frame_rate=3, time_unit='s')
standards.animate_topomap(ch_type='eeg', times=np.arange(0.0,0.6,0.02), frame_rate=3, time_unit='s')
# exact frames are isolated from the times=np.arange structure, frame=rate

##Compare our standard and target ERPs##
evoked_dict = dict(standards = epochs['standards'].average(), targets = epochs['targets'].average())

#colors = dict(Left="Crimson", Right="CornFlowerBlue")
#linestyles = dict(Auditory='-', visual='--')
pick = [evoked_dict["standards"].ch_names.index(i) for i in electrodes]

mne.viz.plot_compare_evokeds(evoked_dict, picks=pick, 
                             ylim=dict(eeg=[plot_max_targets,plot_min_targets]))

# %% ##Difference waveform#
target, standard = epochs["targets"].average, epochs["standards"].average
joint_kwargs = dict(ts_args=dict(time_unit='s'),
                    topomap_args=dict(time_unit='s'))
mne.combine_evoked([targets, standards], weights='equal').plot_joint(**joint_kwargs, title='Targets - Standard', times="peaks")


# %% ##PSD plots - still need to adapt
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
##
# %% ##Event Related Potential/Field Image## includes spectral reordering & lag correction
def order_func(times, data):
    this_data = data[:, (times > 0.4) & (times < 0.900)]
    this_data /= np.sqrt(np.sum(this_data ** 2, axis=1))[:, np.newaxis]
    return np.argsort(spectral_embedding(rbf_kernel(this_data, gamma=1.),
                      n_components=1, random_state=0).ravel())


good_pick = 7  # channel with a clear evoked response
bad_pick = 11  # channel with no evoked response

# We'll also plot a sample time onset for each trial
plt_times = np.linspace(0, .2, len(epochs.events))

plt.close('all')
mne.viz.plot_epochs_image(epochs, [good_pick, bad_pick], sigma=.5,
                          order=order_func, vmin=-50, vmax=50,
                          overlay_times=plt_times, show=True)

# %% ## Plot Events
order = np.arange(raw.info['nchan'])
events = mne.find_events(raw)

mne.viz.plot_events(events, raw.info['sfreq'], raw.first_samp, color=colour,
                    event_id=event_id)
#raw.plot(events=events, n_channels=18, order=order) can scroll through data
#raw.plot_psd();
# %% ##PSD -all channels
raw.plot_psd(fmin=1, fmax=30);
# %% ## Muiltitaper PSD
f, ax = plt.subplots()
psds, freqs = psd_multitaper(epochs, fmin=2, fmax=40, n_jobs=1)
psds = 10. * np.log10(psds)
psds_mean = psds.mean(0).mean(0)
psds_std = psds.mean(0).std(0)

ax.plot(freqs, psds_mean, color='k')
ax.fill_between(freqs, psds_mean - psds_std, psds_mean + psds_std,
                color='k', alpha=.5)
ax.set(title='Multitaper PSD (gradiometers)', xlabel='Frequency',
       ylabel='Power Spectral Density (dB)')
plt.show()

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

