%% Plot ERPs of your epochs
% An ERP is a plot of the raw, usually filtered, data, at one or multiple electrodes. It doesn't use time-frequency data.
% We make ERPs if we have segmented datasets that we want to compare across conditions.

% In this case, you are using all your electrodes, not a subset.
electrodes = {EEG.chanlocs(:).labels};
% Type "electrodes" into the command line. This will show you which number to use for i_chan

% This code will take a grand average of the subjects, making one figure per set.
% This is the normal way to present ERP results.
i_chan = [7];%%%7(Pz) 9(Fz)
for i_set = 1:nsets
    data_out = [];
    exp.setname{i_set}
    % The code below uses a nested loop to determine which segmented dataset corresponds to the right argument in data_out
    % e.g. if you have 5 sets, 20 participants, and 4 events, for i_set ==
    % 2 and i_part == 1 and i_event == 1, the code uses the data from set (2-1)*4*20 + (1-1)*20 + 1 == set 81
    for eegset = 1:nevents
        exp.event_names{1,eegset}
        for i_part = 1:nparts
            
            data_out(:,:,eegset,i_part,i_set) = nanmean(ALLEEG((i_set-1)*nevents*nparts + (eegset-1)*(nparts) + i_part).data,3);
            
            ALLEEG((i_set-1)*nevents*nparts + (eegset-1)*(nparts) + i_part).filename
        end
        
        % this is the averaged ERP data. We take the mean across the selected channels (dim1) and the participants (dim4).
        erpdata(i_set,eegset).cond = squeeze(mean(mean(data_out(i_chan,:,eegset,:,i_set),1),4));
        erpdata_parts(i_set,eegset).cond = squeeze(mean(data_out(i_chan,:,eegset,:,i_set),1));
        all_chan_erpdata(i_set,eegset).cond = squeeze(mean(data_out(:,:,eegset,:,i_set),4));
        %     erpdata = squeeze(mean(mean(data_out(i_chan,:,:,:,i_set),1),4));
        
    end
end

%%
nametags = {'EEG Dervied','Camera Derived'}; %name the rows
%%%%%THE FOLLOWING ARE FOR TONES FOR EACH PARTICIPANT, SEE BELOW FOR MEAN TONES%%%%%
col = ['k','b';'k','b'];
col = ['b','r';'b','r'];
%%%%% Pick your condition 1 = EEG; 2 = Camera%%%%%
cond1 = 2;
cond2 = 2;
%%%%%Pick your tone 1 = low; 2 = high%%%%%
tone1 = 1;
tone2 = 2;

%%%%%ERPs for high/low tones%%%%%
figure;hold on;
for i_part = 1:nparts
    figure;hold on;
% %     subplot(ceil(sqrt(nparts)),ceil(sqrt(nparts)),i_part);
%     subplot(nparts,1,i_part);
    
    boundedline(EEG.times,erpdata_parts(cond1,tone1).cond(:,i_part),std(erpdata_parts(cond1,tone1).cond(:,i_part))./sqrt(length(exp.participants)),col(cond1,tone1),...
        EEG.times,erpdata_parts(cond2,tone2).cond(:,i_part),std(erpdata_parts(cond2,tone2).cond(:,i_part))./sqrt(length(exp.participants)),col(cond2,tone2));
% %     
% % %         subplot(ceil(sqrt(nparts)),ceil(sqrt(nparts)),i_part);boundedline(EEG.times,erpdata_parts(cond1,tone2).cond(:,i_part)-erpdata_parts(cond1,tone1).cond(:,i_part),...
% % %             std(erpdata_parts(cond1,tone2).cond(:,i_part)-erpdata_parts(cond1,tone1).cond(:,i_part))./sqrt(length(exp.participants)),col(cond1,tone2));
% %     %%%%% epoched to last entrainer %%%%%
    xlim([-200 1000])
    ylim([-25  25])
    set(gca,'Ydir','reverse');
    line([0 0],[-15 15],'color','k');
% %     line([300 300],[-10 10],'color','k'); %%%%use this to help find ERP regions
% %     line([550 550],[-10 10],'color','k'); %%%%use this to help find ERP regions
    line([-200 1000],[0 0],'color','k');
    title(['Participant ' exp.participants(i_part)  nametags(cond1)]);
    ylabel('Voltage (mV)')
    xlabel('Time (ms)')
%     xticks([200:100:1300])
%     xticklabels([0:100:1100])
end
 hold off;
 
%% Group Figures

sub_nums = {'003', '004'};
nsubs = length(sub_nums);
figure('Position',[25,25,1000,1000]); 
widthHeight = ceil(sqrt(nsubs));

for i_sub = 1:nsubs 
    subplot(widthHeight,widthHeight,i_sub); 
        boundedline(EEG.times,erpdata_parts(cond1,tone1).cond(:,i_sub),std(erpdata_parts(cond1,tone1).cond(:,i_sub))./sqrt(length(exp.participants)),col(cond1,tone1),...
        EEG.times,erpdata_parts(cond2,tone2).cond(:,i_sub),std(erpdata_parts(cond2,tone2).cond(:,i_sub))./sqrt(length(exp.participants)),col(cond2,tone2));
        xlim([-200 1000])
        ylim([-25  25])
        set(gca,'Ydir','reverse');
        %%%%% epoched to last entrainegfhr %%%%%
        line([0 0],[-15 15],'color','k');
        line([-200 1000],[0 0],'color','k');
        title(['Participant ' exp.participants(i_sub)  nametags(cond1)]);
        ylabel('Voltage (mV)')
        xlabel('Time (ms)')
        hold on
end

 hold off
 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%THE FOLLOWING ARE FOR TONES%%%%%
col = ['b','b';'r','r']; %%%uncomment to compare one tone type across conditions
% % col = ['k','b';'k','r']; %%%uncomment for high and low tones
%%%%% Pick your condition 1 = EEG; 2 = GoPro%%%%%
cond1 = 2;
cond2 = 2;
%%%%%Pick your tone 1 = low; 2 = high%%%%%
tone1 = 1;
tone2 = 2;
%%%%%ERPs for high or low tones%%%%%
% % figure;boundedline(EEG.times(801:2000),erpdata(cond1,tone1).cond(801:2000),std(erpdata_parts(cond1,tone1).cond(801:2000),[],2)./sqrt(length(exp.participants)),col(cond1,tone1),...
% %     EEG.times(801:2000),erpdata(cond2,tone2).cond(801:2000),std(erpdata_parts(cond2,tone2).cond(801:2000),[],2)./sqrt(length(exp.participants)),col(cond2,tone2));

%%%%%Difference Waves for high and low tones%%%%%
figure;boundedline(EEG.times(801:2000),(erpdata(cond1,tone2).cond(801:2000)-erpdata(cond1,tone1).cond(801:2000)),std(erpdata_parts(cond1,tone2).cond(801:2000)-erpdata_parts(cond1,tone1).cond(801:2000),[],2)./sqrt(length(exp.participants)),col(cond1,tone1),...
    EEG.times(801:2000),(erpdata(cond2,tone2).cond(801:2000)-erpdata(cond2,tone1).cond(801:2000)),std(erpdata_parts(cond2,tone2).cond(801:2000)-erpdata_parts(cond2,tone1).cond(801:2000),[],2)./sqrt(length(exp.participants)),col(cond2,tone2));

%%%%% epoched to last entrainer %%%%%
xlim([-200 1000])
ylim([-15 15])
set(gca,'Ydir','reverse');
%%%%% epoched to last entrainer %%%%%
line([0 0],[-15 15],'color','k');
line([-200 1000],[0 0],'color','k');
% line([225 225],[-15 15],'color','k'); %%%%use this to help find ERP regions
% line([300 300],[-15 15],'color','k'); %%%%use this to help find ERP regions
title([  nametags(cond1) 'Difference ERPs']);
ylabel('Voltage (mV)')
xlabel('Time (ms)')


%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%Power Topoplots%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Pick your condition 1 = no headset; 2 = headset%%%%%
cond1 = 1;
cond2 = 2;
%%%%%Pick your tone 1 = low 2 = high%%%%%
tone1 = 1;
tone2 = 2;
%%%%%Pick your time window 300-550 for P3, 175-275 for MMN%%%%%
%%%%%for revised auditory VR with RT, 400-550 for P3, 300-400 for MMN (Eden
%%%%%revised auditory VR for 
%%%%%for revised visual VR with RT, 300-550 for P3, 250-325 for MMN
time1 = 350;
time2 = 550;

time1 = 225;
time2 = 300;

%%%%%Get topoplots for differnce between targets and standards%%%%%
time_window = find(EEG.times>time1,1)-1:find(EEG.times>time2,1)-2;

erp_diff_EEG = (all_chan_erpdata(cond1,tone2).cond-all_chan_erpdata(cond1,tone1).cond);
erp_diff_camera = (all_chan_erpdata(cond2,tone2).cond-all_chan_erpdata(cond2,tone1).cond);

figure('Color',[1 1 1]);
set(gca,'Color',[1 1 1]);
temp1 = mean(erp_diff_EEG(:,time_window),2);
temp2 = mean(erp_diff_camera(:,time_window),2);
temp1(17:18) = NaN;
temp2(17:18) = NaN;

lims = ([-10 10]);
subplot(1,2,1);
topoplot((temp1),exp.electrode_locs, 'whitebk','on','plotrad',.6,'maplimits',lims)
title('EEG');
t = colorbar('peer',gca);
set(get(t,'ylabel'),'String', 'Voltage Difference (uV)');

subplot(1,2,2);
topoplot((temp2),exp.electrode_locs, 'whitebk','on','plotrad',.6,'maplimits',lims)
title('Camera');
t = colorbar('peer',gca);
set(get(t,'ylabel'),'String', 'Voltage Difference (uV)');

%%%%%Difference Topoplots%%%%%
figure('Color',[1 1 1]);
set(gca,'Color',[1 1 1]);

topoplot((temp1-temp2),exp.electrode_locs, 'whitebk','on','plotrad',.6,'maplimits',lims)
title('No Headset - Headset');
t = colorbar('peer',gca);
set(get(t,'ylabel'),'String', 'Voltage Difference (uV)');

