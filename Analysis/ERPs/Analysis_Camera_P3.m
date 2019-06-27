% %% load in datasets
function Analysis_Camera_P3(exp,anal)
sprintf(exp.settings)
try
    %Some lines here to make the variables work in different cases
    
    nparts = length(exp.participants);
    nsets = length(exp.setname);
    nevents = length(exp.events(1,:));
    
    if any(size(exp.event_names) ~= size(exp.events))
        repfactor = size(exp.events)./size(exp.event_names);
        exp.event_names = repmat(exp.event_names, repfactor);
    end
    
% %     if any(size(anal.entrainer_freqs) ~= size(exp.events))
% %         repfactor = size(exp.events)./size(anal.entrainer_freqs);
% %         anal.entrainer_freqs = repmat(anal.entrainer_freqs, repfactor);
% %     end
    
    if isempty(anal.tfelecs) == 1
        anal.tfelecs = exp.brainelecs;
    end
    
    [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
    
    targlatency = cell([nsets,nevents,length(exp.events{1}),nparts]);
    event_trials = cell([nsets,nevents,length(exp.events{1}),nparts]);
    all_event_phase = cell([nsets,nevents,nparts,anal.tfelecs(end)]);
    
    %% Load the data
    %The main loop loops over events, then participants, then sets.
    for i_set = 1:nsets
% %         if i_set == 2
% %             if exp.prepross == 3
% %                 exp.name = 'auditory_hs';
% %             elseif exp.prepross == 6
% %                 exp.name = 'visual_hs';
% %             end
% %         end
        %     exp.setname(i_set)
        tic
        %     for i_part = 1:nparts
        %         sprintf(['Loading Participant ' num2str(exp.participants{i_part}) '...' ])
        
        
        nevents = length(exp.events(i_set,:));
        %         for i_event = 1:nevents
        %%%%%ADDED CODE%%%%%%
        Corrected = {'uncorrected','corrected'};

        for i_event = 1:nevents
            for i_part=1:nparts
                sprintf(['Loading Condition ' num2str(exp.setname{i_set}) '...' ])
                sprintf(['Loading Event ' num2str(exp.event_names{1,i_event}) '...' ])
                sprintf(['Loading Participant ' num2str(exp.participants{i_part}) '...' ])
                %%%%%%%%%%%%%%%%%%%%%%%%
                filename = [exp.name '_' exp.settings '_' exp.participants{i_part} '_' exp.setname{i_set} '_' exp.event_names{i_set,i_event} '_' Corrected{exp.corrected} ] %
%                 if exp.old_vr == 0
%                     filename_temp = ['Segments_' exp.name '_' exp.settings '_' exp.participants{i_part} '_' exp.setname{i_set} '_' exp.event_names{i_set,i_event} '.set']
%                     if exist([exp.pathname  '\' exp.settings '\Segments\' exp.setname{i_set} '\' filename_temp],'file') == 0
%                         disp('This condition does not exist, going to the next iteration.');
%                         pause(0.5);
%                         continue
%                     end
%                 end
                % Load the Time frequency data, if needed.
                if strcmp('on',anal.tf) == 1 % only load these variables if we are loading time-frequency data
                    ersp(i_part,i_set,i_event,:,:,:) = struct2array(load([exp.pathname '\' exp.settings '\TimeFrequency\' 'TF_' filename '.mat'],'ersp'));
                    itc(i_part,i_set,i_event,:,:,:) = struct2array(load([exp.pathname '\' exp.settings '\TimeFrequency\' 'TF_' filename '.mat'],'itc'));
                    if i_part == 1 && i_set == 1 && i_event == 1
                        times = struct2array(load([exp.pathname '\' exp.settings '\TimeFrequency\' 'TF_' filename '.mat'],'times'));
                        freqs = struct2array(load([exp.pathname '\' exp.settings '\TimeFrequency\' 'TF_' filename '.mat'],'freqs'));
                    end
                end
                
                % Load the EEGLAB datasets, if needed.
                if strcmp('on',anal.segments) == 1  % only load these variables if we are loading ERP data
                    try %Unfortunately, this load procedure can break sometimes in a non-reproducible way. So if an error happens here, we wait half a second and try again.
                        EEG = pop_loadset('filename',['Segments_' filename '.set'],'filepath',[exp.pathname  '\' exp.settings '\Segments\' exp.setname{i_set} '\']);
                        [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
                    catch
                        pause(.5)
                        EEG = pop_loadset('filename',['Segments_' filename '.set'],'filepath',[exp.pathname  '\' exp.settings '\Segments\' exp.setname{i_set} '\']);
                        [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
                    end
                elseif strcmp('on',anal.singletrials) == 1 %For single trial data, we want to load it, but saving it is unnecessary and space intensive
                    try %Unfortunately, this load procedure can break sometimes in a non-reproducible way. So if an error happens here, we wait half a second and try again.
                        EEG = pop_loadset('filename',['Segments_' filename '.set'],'filepath',[exp.pathname  '\' exp.settings '\Segments\' exp.setname{i_set} '\']);
                    catch
                        pause(.5)
                        EEG = pop_loadset('filename',['Segments_' filename '.set'],'filepath',[exp.pathname  '\' exp.settings '\Segments\' exp.setname{i_set} '\']);
                    end
                elseif strcmp('on',anal.tf) == 1 % if we are loading time-frequency data only, then we just need one of these.
                    if i_part == 1 && i_set == 1 && i_event == 1
                        EEG = pop_loadset('filename',['Segments_' filename '.set'],'filepath',[exp.pathname  '\' exp.settings '\Segments\' exp.setname{i_set} '\']);
                        [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
                    end
                end
                
                % Load the Single Trial complex values, if needed
                if strcmp('on',anal.singletrials) == 1 % only load these variables if we are loading single trial data
                    ntrigs = length(exp.events{i_set});
                    if i_part == 1 && i_set == 1 && i_event == 1 && strcmp('on',anal.tf) == 1
                        times = struct2array(load([exp.pathname '\' exp.settings '\TimeFrequency\' 'TF_' filename '.mat'],'times'));
                        freqs = struct2array(load([exp.pathname '\' exp.settings '\TimeFrequency\' 'TF_' filename '.mat'],'freqs'));
                    end
                    
                    %This block finds the latency of each event listed in exp.events, and the trials it appeared on.
                    for nperevent = 1:ntrigs
                        for i_trial = 1:EEG.trials
                            if any(strcmp(num2str(exp.events{i_set,i_event}(nperevent)),EEG.epoch(i_trial).eventtype)) == 1
                                targlatency{i_set,i_event,nperevent,i_part} = [targlatency{i_set,i_event,nperevent,i_part} EEG.epoch(i_trial).eventlatency(find(strcmp(num2str(exp.events{i_set,i_event}(nperevent)),EEG.epoch(i_trial).eventtype)))];
                                event_trials{i_set,i_event,nperevent,i_part} = [event_trials{i_set,i_event,nperevent,i_part} i_trial];
                            end
                        end
                    end
                    
                    for i_chan = anal.singletrialselecs
                        try %Unfortunately, this load procedure can break sometimes in a non-reproducible way. So if an error happens here, we wait half a second and try again.
                            channeldata = load([exp.pathname '\' exp.settings '\SingleTrials\' exp.participants{i_part} '_' exp.setname{i_set} '_' exp.event_names{i_set,i_event} '\' EEG.chanlocs(i_chan).labels '_SingleTrials_' filename '.mat']);
                            tf_data{i_set,i_event,i_part,i_chan} = channeldata.elec_tf_data;
                            %                         tf_power{i_set,i_event,i_part,i_chan} = abs(channeldata.elec_tf_data);
                            %                         tf_phase{i_set,i_event,i_part,i_chan} = angle(channeldata.elec_tf_data)+pi;
                            tf_trialnum{i_set,i_event,i_part,i_chan} = channeldata.elec_tf_trialnum;
                        catch
                            pause(.5)
                            channeldata = load([exp.pathname '\' exp.settings '\SingleTrials\' exp.participants{i_part} '_' exp.setname{i_set} '_' exp.event_names{i_set,i_event} '\' EEG.chanlocs(i_chan).labels '_SingleTrials_' filename '.mat']);
                            tf_data{i_set,i_event,i_part,i_chan} = channeldata.elec_tf_data;
                            tf_trialnum{i_set,i_event,i_part,i_chan} = channeldata.elec_tf_trialnum;
                        end
                        
                        if i_part == 1 && i_set == 1 && i_event == 1
                            times = channeldata.times;
                            freqs = channeldata.freqs;
                        end
                        
                        %Here we find the complex values associated with each event in exp.events
                        %the time is determined by the event latency, and the frequency is selected in the wrapper.
                        if isempty(event_trials{i_set,i_event,nperevent,i_part}) ~= 1
                            entr_freq = find(freqs>anal.entrainer_freqs(i_set,i_event),1);
                            pre_targ_time = find(times < mode(cell2mat(targlatency{i_set,i_event,nperevent,i_part})),1,'last');
                            
                            current_complex = [squeeze( tf_data{i_set,i_event,i_part,i_chan}(entr_freq,pre_targ_time,event_trials{i_set,i_event,nperevent,i_part}) )]'; %This is the complex of every trial matching the current event
                            current_phase = angle(current_complex)'; %The takes just the phase (imaginary) component of the complex
                            
                            all_event_phase{i_set,i_event,i_part,i_chan} = [all_event_phase{i_set,i_event,i_part,i_chan}; current_phase];
                            
                            [circ_mean_v,range_v,X,Y,cos_a,sin_a] = circle_mean(circ_rad2ang(current_phase));    % Computes the circular mean for each condition in each subject
                            phase_out(i_set,i_part,i_event,i_chan) = circ_mean_v;                        %record the condition circular phase mean
                            range_out(i_set,i_part,i_event,i_chan) = range_v;                            %record the range (concentration) of each of these means
                            X_out(i_set,i_part,i_event,i_chan) = X;                                    %record the cosine component of the mean
                            Y_out(i_set,i_part,i_event,i_chan) = Y;                                    %record the sine component of the mean
                            cos_out(i_set,i_part,i_event,i_chan) = cos_a;
                            sin_out(i_set,i_part,i_event,i_chan) = sin_a;
                        end
                        
                    end
                end
            end
        end
        toc
    end
    
    A = who;
    for i = 1:length(A)
        assignin('base', A{i}, eval(A{i}));
    end
    
catch ME
    A = who;
    for i = 1:length(A)
        assignin('base', A{i}, eval(A{i}));
    end
    throw(ME)
end
end