%% PREPROCESSING AND ANALYSIS WRAPPER
%clear and close everything
ccc

%% Settings for loading the raw data
exp.preprocess = 3; %%%set to 1 for EEG, 2 for Camera, 3 for LOADING PROPROCESSED DATA
exp.uncorrected = 0; %%%do you want to use the uncorrected camera times?
% % % 0 = corrected/shifted 
% % % 1 = aligned, uncorrected/unshifted, 
% % % 2 = not aligned, uncorrected/unshifted 
% % % 3 = aligned, shifted, uncorrected 
% % % 4 = aligned, corrected, unshifted

exp.participants = {'003';'004';'005';'006';'007';'008';'009';'010';'011'};
exp.participants = {'003';'004';'005';'007';'008';'009';'010';'011'};
% % exp.participants = {'010';'011'};

if exp.preprocess == 1
    %Datafiles must be in the format exp_participant, e.g. EEGexp_001.vhdr
    exp.name = 'camera_p3';
    
    exp.pathname = 'M:\Experiments\Visual P3\EEG_Data';
    
    exp.events = {[1],[2]};    %must be matrix (sets x events)
    exp.event_names = {'Low_Tones','High_Tones'}; %name the columns

    exp.setname = {'EEG_Latencies'}; %name the rows
    exp.selection_cards = {'1','2'};
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif exp.preprocess == 2
    %Datafiles must be in the format exp_participant, e.g. EEGexp_001.vhdr
    exp.name = 'camera_p3';

    exp.pathname = 'M:\Experiments\Visual P3\EEG_Data';
    
    exp.events = {[5],[6]};    %must be matrix (sets x events)
    exp.event_names = {'Low_Tones','High_Tones'}; %name the columns

    exp.setname = {'Camera_Latencies'}; %name the rows
    exp.selection_cards = {'5','6'};
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif exp.preprocess == 3
    %Datafiles must be in the format exp_participant, e.g. EEGexp_001.vhdr
    exp.name = 'camera_p3';
    
    exp.pathname = 'M:\Experiments\Visual P3\EEG_Data';
    
    exp.events = {[1],[2];...
        [5],[6]};    %must be matrix (sets x events)
    exp.event_names = {'Low_Tones','High_Tones'}; %name the columns

    exp.setname = {'EEG_Latencies','Camera_Latencies'}; %name the rows
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

% Each item in the exp.events matrix will become a seperate dataset, including only those epochs referenced by the events in that item.
%e.g. 3 rows x 4 columns == 12 datasets/participant

% The settings will be saved as a new folder. It lets you save multiple datasets with different preprocessing parameters.
exp.settings = 'Camera_P3';

%% Preprocessing Settings
%segments settings
exp.segments = 'on'; %Do you want to make new epoched datasets? Set to "off" if you are only changing the tf settings.
%Where are your electrodes? (.ced file)
exp.electrode_locs = 'M:\Analysis\Muse\eeg_analysis\16chanmuse.ced';

%% Filter the data?
exp.filter = 'on';
exp.lowpass = 50;
exp.highpass = 0;

%% Re-referencing the data
exp.refelec = 16; %which electrode do you want to re-reference to?
exp.brainelecs = [1:15]; %list of every electrode collecting brain data (exclude mastoid reference, EOGs, HR, EMG, etc.

%% Epoching the data
%Choose what to epoch to. The default [] uses every event listed above.
%Alternatively, you can epoch to something else in the format {'TRIG'}. Use triggers which are at a consistent time point in every trial.
exp.epochs = [];
exp.epochslims = [-1 1]; %Tone Onset in seconds; epoched trigger is 0 e.g. [-1 2]
% % exp.epochslims = [-1 2.5]; %Picture Onset in seconds; epoched trigger is 0 e.g. [-1 2]
exp.epochbaseline = [-200 0]; %remove the for each epoched set, in ms. e.g. [-200 0]

%% Artifact rejection.
% Choose the threshold to reject trials. More lenient threshold followed by an (optional) stricter threshold
exp.preocularthresh = [-1000 1000]; %First happens before the ocular correction.
exp.postocularthresh = [-500 500]; %Second happens after. Leave blank [] to skip

%% Blink Correction
%the Blink Correction wants dissimilar events (different erps) seperated by commas and similar events (similar erps) seperated with spaces. See 'help gratton_emcp'
% % exp.selection_cards = {'1','2'};
%%%%

%% Time-Frequency settings
%Do you want to run time-frequency analyses? (on/off)
exp.tf = 'off';
%Do you want to save the single-trial data? (on/off) (Memory intensive!!!)
exp.singletrials = 'off';
%Do you want to use all the electrodes or just a few? Leave blank [] for all (will use same as exp.brainelecs)
exp.tfelecs = [];
%Saving the single trial data is memory intensive. Just use the electrodes you need.
exp.singletrialselecs = [3];

%% Wavelet settings
%how long is your window going to be? (Longer window == BETTER frequency resolution & WORSE time resolution)
exp.winsize = 512; %in ms; use numbers that are 2^x, e.g. 2^10 == 1024ms
%baseline will be subtracted from the power variable. It is relative to your window size.
exp.erspbaseline = [-200 0]; %e.g., [-200 0] will use [-200-exp.winsize/2 0-exp.winsize/2]; Can use just NaN for no baseline
%Instead of choosing a windowsize, you can choose a number of cycles per frequency. See "help popnewtimef"
exp.cycles = [0]; %leave it at 0 to use a consistent time window
exp.freqrange = [1 40]; % what frequencies to consider? default is [1 50]
%%%%

%% Save your pipeline settings
save([exp.settings '_Settings'],'exp') %save these settings as a .mat file. This will help you remember what settings were used in each dataset

% % Run Preprocessing
if exp.preprocess == 3 
    disp('Make sure the variable ''exp.prepross'' is set to ''1'' or ''2'' if you want to preprocess the data.');
else
    Preprocessing_Camera_P3(exp) %comment out if you're only doing analysis
end

%% Run Analysis
%Don't want to change all the above settings? Load the settings from the saved .mat file.

%choose the data types to load into memory (on/off)
anal.segments = 'on'; %load the EEG segments?
anal.tf = 'off'; %load the time-frequency data?

anal.singletrials = 'off'; %load the single trial data?
anal.entrainer_freqs = [20; 15; 12; 8.5; 4]; %Single trial data is loaded at the event time, and at the chosen frequency.

anal.tfelecs = []; %load all the electodes, or just a few? Leave blank [] for all.
anal.singletrialselecs = [2 3 4 6];

if exp.preprocess ~= 3
    disp('Make sure variable ''exp.prepross'' is set to ''3'' if you want to perform data analysis.');
else
    Analysis_Camera_P3(exp,anal) % The Analysis primarily loads the processed data. It will attempt to make some figures, but most analysis will need to be done in seperate scripts.
end
