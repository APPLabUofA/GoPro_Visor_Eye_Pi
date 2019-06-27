function Preprocessing_Camera_P3(exp)

try
    
    %     try
    %         parpool
    %     catch
    %     end
    
    nparts = length(exp.participants);
    nsets = length(exp.setname);
    
    if any(size(exp.event_names) ~= size(exp.events))
        repfactor = size(exp.events)./size(exp.event_names);
        exp.event_names = repmat(exp.event_names, repfactor);
    end
    
    if isempty(exp.epochs) == 1
        exp.epochs = cellstr(num2str(cell2mat( reshape(exp.events,1,size(exp.events,1)*size(exp.events,2)) )'))';
    end
    
    %initialize EEGLAB
    [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
    %subject numbers to analyze
    
    %% Load data and channel locations
    if strcmp('on',exp.segments) == 1
        
        for i_part = 1:nparts
            sprintf(['Processing Participant ' num2str(exp.participants{i_part})])
            
            %% load a data file
            EEG = pop_loadbv(exp.pathname, [exp.participants{i_part} '_' exp.name '.vhdr']);
            
            % load channel information
            EEG=pop_chanedit(EEG, 'load',{exp.electrode_locs 'filetype' 'autodetect'});
            
            %% Filter the data with low pass of 30
            if strcmp('on',exp.filter) == 1
                EEG = pop_eegfilt( EEG, exp.highpass, exp.lowpass, [], 0);
            end
            
            %% arithmetically rereference to linked mastoid (M1 + M2)/2
            for x=exp.brainelecs
                EEG.data(x,:) = (EEG.data(x,:)-((EEG.data(exp.refelec,:))*.5));
            end
            
            %change markers so they can be used by the gratton_emcp script
            
            allevents = length(EEG.event);
            correct_rt = 0;
            incorrect_rt = 0;
            
            
            if exp.preprocess == 1
                for i_event = 1:allevents
                    if EEG.event(i_event).type == "S  1"
                        EEG.event(i_event).type = '1';
                    elseif EEG.event(i_event).type == "S  2"
                        EEG.event(i_event).type = '2'; 
                    end
                end         
            end
            
            if exp.preprocess == 2 
                if exp.corrected == 1%%%aligned and corrected, and shifted
                    T = readtable(strcat('M:\Data\GoPro_Visor\Experiment_1\Video_Times\Dataframe_df3_whole_final_', exp.participants{i_part} ,'.csv'));
                    gopro_times = table2array(T(1:height(T),2:3));
                    gopro_times(:,1) = round(gopro_times(:,1).*1000);

                    EEG.event_2 = [];
                    start_offset = EEG.event(2).latency;

                    for i_event = 1:length(gopro_times)
                        if gopro_times(i_event,2) == 1
                            EEG.event_2(i_event).type = '5';
                        elseif gopro_times(i_event,2) == 2
                            EEG.event_2(i_event).type = '6';
                        end
                        EEG.event_2(i_event).latency = gopro_times(i_event,1) + start_offset;

                    end
                    EEG.event = EEG.event_2; % replace the bv event stream with the gopro once transformed and aligned via difference of 1st and 2nd trig
                    allevents = length(EEG.event);
                    
                    
                elseif exp.corrected == 2%%%aligned and corrected, and shifted
                    T = readtable(strcat('M:\Data\GoPro_Visor\Experiment_1\Video_Times\Dataframe_df3_whole_final_', exp.participants{i_part} ,'.csv'));
                    gopro_times = table2array(T(1:height(T),[14,3]));
                    gopro_times(:,1) = round(gopro_times(:,1).*1000);
                    
                    EEG.event_2 = [];
                    start_offset = EEG.event(2).latency;

                    for i_event = 1:length(gopro_times)
                        if gopro_times(i_event,2) == 1
                            EEG.event_2(i_event).type = '5';
                        elseif gopro_times(i_event,2) == 2
                            EEG.event_2(i_event).type = '6';
                        end
                        EEG.event_2(i_event).latency = gopro_times(i_event,1) + start_offset;

                    end
                    EEG.event = EEG.event_2; % replace the bv event stream with the gopro once transformed and aligned via difference of 1st and 2nd trig
                    allevents = length(EEG.event);
                end
            
            end
            
            %% The triggers are early
            % %             [EEG] = VpixxEarlyTriggerFix(EEG);
            
            %% Extract epochs of data time locked to event
            %Extract data time locked to targets and remove all other events
            EEG = pop_epoch( EEG, exp.epochs, exp.epochslims, 'newname', [exp.participants{i_part} '_epochs'], 'epochinfo', 'yes');
            %subtract baseline
            EEG = pop_rmbase( EEG, exp.epochbaseline);
            
            %% Artifact Rejection, Correction, then 2nd Rejection
            
            %Artifact rejection, trials with range >exp.preocularthresh uV
            if isempty(exp.preocularthresh) == 0
                rejtrial = struct([]);
                EEG = pop_eegthresh(EEG,1,[1:size(EEG.data,1)],exp.preocularthresh(1),exp.preocularthresh(2),EEG.xmin,EEG.xmax,0,1);
                rejtrial(i_part,1).ids = find(EEG.reject.rejthresh==1);
            end
            
            %EMCP occular correction
            temp_ocular = EEG.data(end-1:end,:,:); %to save the EYE data for after
            EEG = gratton_emcp(EEG,exp.selection_cards,{'VEOG'},{'HEOG'}); %this assumes the eye channels are called this
            EEG.emcp.table %this prints out the regression coefficients
            EEG.data(end-1:end,:,:) = temp_ocular; %replace the eye data
            
            %Baseline again since this changed it
            EEG = pop_rmbase( EEG, exp.epochbaseline);
            
            %Artifact rejection, trials with range >exp.postocularthresh uV
            if isempty(exp.postocularthresh) == 0
                EEG = pop_eegthresh(EEG,1,[1:size(EEG.data,1)-2],exp.postocularthresh(1),exp.postocularthresh(2),EEG.xmin,EEG.xmax,0,1);
                rejtrial(i_part,2).ids = find(EEG.reject.rejthresh==1);
            end
            
            %% Event coding
            
            %             here we rename the event codes to the event names in the wrapper so stimuli can be easily identified later
            for i_trial = 1:EEG.trials
                for i_set = 1:nsets
                    nevents = length(exp.events(i_set,:));
                    for i_event = 1:nevents
                        nperevent = length(exp.events{i_set,i_event});
                        for j_event = 1:nperevent
                            if strcmp(num2str(exp.events{i_set,i_event}(j_event)),EEG.epoch(i_trial).eventtype) ==1;
                                EEG.epoch(i_trial).eventcode = exp.event_names(i_set,i_event);
                            end
                        end
                    end
                end
            end
            
            %here we collect the triggers for each trial that matched an event
            output = struct([]);
            output(i_part).target(EEG.trials) = 0
            output(i_part).targetlatency(EEG.trials) = 0
            for i_trial = 1:EEG.trials
                for i_set = 1:nsets
                    for i_event = 1:length(exp.events(i_set,:));
                        if isempty(find(strcmp(exp.event_names(i_set,i_event),EEG.epoch(i_trial).eventcode)== 1 ,1)) == 0
                            %                             output(i_part).target(i_trial) = str2num( EEG.epoch(i_trial).eventtype );
                            %                             output(i_part).targetlatency(i_trial) = EEG.epoch(i_trial).eventlatency;
                        end
                    end
                end
            end
            
            %here we make the trial type and latency a NaN if no trial matched an event
            output(i_part).target(output(i_part).target == 0) = NaN;
            output(i_part).targetlatency(output(i_part).targetlatency == 0) = NaN;
            
            %here we replace rejected trials with NaNs as well
            if isempty(exp.postocularthresh) == 0
                for i_trial = rejtrial(i_part,2).ids
                    if i_trial > 50
                        output(i_part).target = [output(i_part).target(1:i_trial-1), NaN, output(i_part).target(i_trial:end)];
                        output(i_part).targetlatency = [output(i_part).targetlatency(1:i_trial-1), NaN, output(i_part).targetlatency(i_trial:end)];
                    else
                        output(i_part).target = [output(i_part).target(1:i_trial-1), 2, output(i_part).target(i_trial:end)];
                        output(i_part).targetlatency = [output(i_part).targetlatency(1:i_trial-1), NaN, output(i_part).targetlatency(i_trial:end)];
                    end
                end
            end
            
            if isempty(exp.preocularthresh) == 0
                for i_trial = rejtrial(i_part,1).ids
                    if i_trial > 50
                        output(i_part).target = [output(i_part).target(1:i_trial-1), NaN, output(i_part).target(i_trial:end)];
                        output(i_part).targetlatency = [output(i_part).targetlatency(1:i_trial-1), NaN, output(i_part).targetlatency(i_trial:end)];
                    else
                        output(i_part).target = [output(i_part).target(1:i_trial-1), 2, output(i_part).target(i_trial:end)];
                        output(i_part).targetlatency = [output(i_part).targetlatency(1:i_trial-1), NaN, output(i_part).targetlatency(i_trial:end)];
                    end
                end
            end
            
            %here we save a variable into the EEG file with the original trial numbers, with NaNs for rejected trials as well as trials that matched no events
            EEG.replacedtrials.target = output(i_part).target;
            EEG.replacedtrials.targetlatency = output(i_part).targetlatency;
            
            for i_set = 1:nsets
                disp(['Saving: ' exp.setname{i_set}]);
                
                if ~exist([exp.pathname '\' exp.settings '\'],'dir')
                    mkdir([exp.pathname '\' exp.settings '\']);
                end
                
                if ~exist([exp.pathname '\' exp.settings '\Segments\' exp.setname{i_set} '\'],'dir')
                    mkdir([exp.pathname '\' exp.settings '\Segments\' exp.setname{i_set} '\']);
                end
                %% Select individual events and Save
                
                Corrected = {'uncorrected','corrected'};
                
                EEG.exp = exp;
                setEEG = EEG;   %replace the stored data with this new set
                nevents = length(exp.events(i_set,:));
                for i_event = 1:nevents
                    filename = [exp.name '_' exp.settings '_' exp.participants{i_part} '_' exp.setname{i_set} '_' exp.event_names{i_set,i_event} '_' Corrected{exp.corrected}]
                    eventEEG = pop_selectevent(setEEG, 'type', exp.events{i_set,i_event}, 'latency', ['0 <= ' num2str(exp.epochslims(2)*1000)], 'deleteevents','off','deleteepochs','on','invertepochs','off');
                    eventEEG = pop_saveset(eventEEG, 'filename',['Segments_' filename '.set'],'filepath',[exp.pathname  '\' exp.settings '\Segments\' exp.setname{i_set} '\'])
                end
            end
            
        end
    end
    
    
    %% Time-Frequency Analysis
    % load in datasets
    exp.erspbaseline = exp.erspbaseline - exp.winsize/2
    if isempty(exp.tfelecs) == 1
        exp.tfelecs = exp.brainelecs;
    end
    if strcmp('on',exp.tf) == 1 || strcmp('on',exp.singletrials) == 1
        for i_set = 1:nsets
            if ~exist([exp.pathname '\' exp.settings '\TimeFrequency\' exp.setname{i_set} '\'])
                mkdir([exp.pathname '\' exp.settings '\TimeFrequency\' exp.setname{i_set} '\']);
            end
            nevents = length(exp.events(i_set,:));
            
            for i_part = 1:nparts
                sprintf(['Participant ' num2str(exp.participants{i_part})])
                for i_event = 1:nevents
                    
                    if ~exist([exp.pathname '\' exp.settings '\SingleTrials\' exp.participants{i_part} '_' exp.setname{i_set} '_' exp.event_names{i_set,i_event} '\'])
                        mkdir([exp.pathname '\' exp.settings '\SingleTrials\' exp.participants{i_part} '_' exp.setname{i_set} '_' exp.event_names{i_set,i_event} '\']);
                    end
                    
                    filename = [exp.name '_' exp.settings '_' exp.participants{i_part} '_' exp.setname{i_set} '_' exp.event_names{i_set,i_event}]
                    EEG = pop_loadset('filename',['Segments_' filename '.set'],'filepath',[exp.pathname  '\' exp.settings '\Segments\' exp.setname{i_set} '\']);
                    
                    if strcmp('on',exp.tf) == 1 && strcmp('on',exp.singletrials) == 1
                        tfchannels = union(exp.tfelecs,exp.singletrialselecs)
                    elseif strcmp('on',exp.tf) == 1 && strcmp('on',exp.singletrials) == 0
                        tfchannels = exp.tfelecs
                    elseif strcmp('on',exp.tf) == 0 && strcmp('on',exp.singletrials) == 1
                        tfchannels = exp.singletrialselecs
                    end
                    
                    if i_set == 1
                        if i_part == 1
                            if i_event == 1
                                for i_chan = tfchannels(end)
                                    [ersp(i_chan,:,:),itc(i_chan,:,:),powbase,times,freqs,dum1,dum2,tf_data(i_chan).trials]  = pop_newtimef( EEG, 1, i_chan, exp.epochslims*1000, exp.cycles , ...
                                        'topovec', i_chan, 'elocs', EEG.chanlocs, 'chaninfo', EEG.chaninfo, 'caption', 'Pz', 'baseline', exp.erspbaseline, 'freqs', exp.freqrange, 'freqscale', 'linear', ...
                                        'padratio', 4,'plotphase','off','plotitc','off','plotersp','off','winsize',exp.winsize,'timesout',400);
                                end
                            end
                        end
                    end
                    
                    for i_chan = tfchannels
                        [ersp(i_chan,:,:),itc(i_chan,:,:),powbase,times,freqs,dum1,dum2,tf_data(i_chan).trials]  = pop_newtimef( EEG, 1, i_chan, exp.epochslims*1000, exp.cycles , ...
                            'topovec', i_chan, 'elocs', EEG.chanlocs, 'chaninfo', EEG.chaninfo, 'caption', 'Pz', 'baseline', exp.erspbaseline, 'freqs', exp.freqrange, 'freqscale', 'linear', ...
                            'padratio', 4,'plotphase','off','plotitc','off','plotersp','off','winsize',exp.winsize,'timesout',400);
                    end
                    
                    if strcmp('on',exp.tf) == 1;
                        save([exp.pathname '\' exp.settings '\TimeFrequency\' 'TF_' filename '.mat'],'ersp','itc','times','freqs','powbase','exp')
                    end
                    
                    if strcmp('on',exp.singletrials) == 1;
                        for i_chan = exp.singletrialselecs
                            elec_tf_data = tf_data(i_chan).trials;
                            elec_tf_trialnum = [];
                            for nperevent = 1:length(exp.events{i_set,i_event});
                                elec_tf_trialnum = [elec_tf_trialnum find(EEG.replacedtrials.target==exp.events{i_set,i_event}(nperevent))];
                            end
                            size_check(i_part,i_set,i_chan,:)=([size(elec_tf_data) size(elec_tf_trialnum)]);
                            save([exp.pathname '\' exp.settings '\SingleTrials\' exp.participants{i_part} '_' exp.setname{i_set} '_' exp.event_names{i_set,i_event} '\' EEG.chanlocs(i_chan).labels '_SingleTrials_' filename '.mat'],'elec_tf_data','elec_tf_trialnum','times','freqs','powbase','exp')
                        end
                    end
                end
                
            end
            eeglab redraw
        end
    end
    
catch ME
    A = who;
    for i = 1:length(A)
        assignin('base', A{i}, eval(A{i}));
    end
    throw(ME)
end
end