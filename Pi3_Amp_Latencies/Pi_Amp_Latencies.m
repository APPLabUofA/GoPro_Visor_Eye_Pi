
%%%Now we will go through the EEG file and determine our latencies%%%
filepath = ['M:\Experiments\Visual P3\EEG_Data'];
filename = ['camera_test_5.vhdr'];

[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
EEG = pop_loadbv(filepath, filename, [], []);
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'setname','timingtest','gui','off');

start_trigger = ALLEEG(1).event(2).latency;
EEG_latencies = zeros(1,(length(ALLEEG(1).event)-3));

for i_event = 3:(length(ALLEEG(1).event)-1)
    
    EEG_latencies(i_event-2) = (ALLEEG(1).event(i_event).latency - start_trigger)/ALLEEG.srate;
    
end

%%%now let's load the times recorded by the pi%%%
%%%need to subtract 5 from these since there is 5 seconds before
%%%the red LEDs, indicating the start, are turned on%%%
pi_recorded_times = csvread('M:\Experiments\Visual P3\EEG_Data\000_05_all_visual_p3_trigs_amp.csv',1,0,[1,0,1,249]);
pi_recorded_times = pi_recorded_times - 5;



all_latencies(1,:) = EEG_latencies;
all_latencies(2,:) = pi_recorded_times;

conditions = {'EEG','Pi Times'};

figure;hold on;
colours = ['k','b'];

for i_plot = 1:2
    
    plot(all_latencies(i_plot,:),[1:250],'color',colours(i_plot));
    
end

xlabel('Time (Seconds)');ylabel('Trial');legend('EEG','Pi Times');
hold off;

%%%now let's plot each of our difference latencies%%%
conditions = {'EEG - Pi Times'};

figure;hold on;
colours = ['k','b'];

plot(all_latencies(1,:)-all_latencies(i_plot,:),[1:250],'color',colours(i_plot));

xlabel('Time (Seconds)');ylabel('Trial');legend('EEG - Pi Times');
hold off;



