%%%first we will read each of the video files%%%

video_name_gp = 'M:\Experiments\Visual P3\Videos\003_camera_p3.MP4';
video_gp_ref = VideoReader(video_name_gp);
gp_frames = video_gp_ref.NumberOfFrames;

%%%%%since matlab seems to have issues loading a minute of video, scan
%%%%%through the video to find the approximate time when and LED flash
%%%%%occurs, and load several second before and after that flash%%%

%%%since we only care about when the LEDs change%%%
%%%we only look at certain pixels and discard the rest%%%
%%%these values were determined by looking at a single frame%%%
%%%from the video and manually determining which pixels contain an LED%%%

%%%our video files, when imported, are 4 dimensional variables%%%
%%%gp_video(y_dim,x_dim,rgb,frames)

%%%first determine when the EEG recording was started, then determine when
%%%he lights/chamber were completely black (do not need to be terribly
%%%accurate with this). Can then use these values to find when the first
%%%LED flash occurred

%%%can check to see where the first flash occurs and%%%
%%%use this to determine which section of each frame to focus on%%%

%%%load only a small chunk of video in order to locate the LED position%%%
%%%to determine the appropriate frame to load, multiply the time (in
%%%seconds) of the video by 240 (or whatever you have the FPS set to)%%%
%%%add the start and stop frames to the square [+500, -500] brackets below%%%
start_eeg = 1442;
start_search = 9000;

gp_video_test = read(video_gp_ref,[start_search,10000]);
figure;imagesc(gp_video_test(:,:,:,1));

%%%so, to find when the LED flash occurs, determine when the lightsi n the
%%%chamber are completely off and then use the bleow command to find when
%%%the frame is no longer completely black

flash_frame = find(mean(mean(mean(gp_video_test(:,:,[2,3],:),1),2),3) > 0.1,1)

start_flash = flash_frame + start_search - 1;
%%%now plot the frame number that the above code returns. If it return and
%%%empty matrix (basically indicating that it didn't find a flash) you may
%%%need to load a different chunk of data%%%
gp_video_test = read(video_gp_ref,[1,2000]);
imagesc(gp_video_test(:,:,:,1));

testing = read(video_gp_ref,[9616,9616]);
figure;imagesc(testing(:,:,:,1));

%%%determine the x and y coordinates of the LEDs and enter them below%%%

% % % %%%find the mean rgb values of a single frame%%%
% % % mean(mean(gp_video_test(:,:,:,13475),1),2)
% % % 
% % % %%%can use this to try and find when the lights go out, and LEDs (find out n-1 for frame)%%%
% % % find(mean(mean(gp_video_test(:,:,2,[12960:gp_frame_end]),1),2) > 0.1,1)
% % % find(mean(mean(gp_video_test(:,:,2,[1:end]),1),2) > 0.1,1)
% % % 
% % % %%%plot 50 frames at once%%%
% % for i_figure = 1:50
% %     begin_frame = 4800 + ((i_figure-1)*25);
% %     figure;hold on;
% %     for i_frame = 1:25
% %         subplot(5,5,i_frame);imagesc(gp_video_test(:,:,:,begin_frame+i_frame));
% %         title(['Frame ' num2str(begin_frame + i_frame)]);
% %     end
% %     hold off;
% % end

% % figure;
% % imagesc(gp_video_test([280:380],[320:500],:,10456));
% %
% % mean(mean(gp_video_test([280:380],[320:500],:,10456)));
% %
% % %%%and now find when the LEDs turned on
% % find(mean(mean(mean(gp_video_test(:,:,:,[door_closed:11040]),1),2),3) > 10,1);
% % start_flash = 8897;
% % mean(mean(mean(gp_video_test(:,:,:,8897))))
% % figure;
% % imagesc(gp_video_test(:,:,:,8897));

gp_video_test = [];

%%%for video 0073%%%
start_eeg = 4274;
off_flash = 5133;
gp_x = [280:310];
gp_y = [290:320];

%%%for 003 (video 0079)%%%
%%%lights are off at frame 3999
%%%mean value of black circle is about 52.2006
start_eeg = 652;
door_closed = 4800;
start_flash = 8897;
gp_x = [320:500];
gp_y = [280:380];

%%%for 004 (video 0080)%%%
%%%lights are off at frame 3999
%%%mean value of black circle is about 52.2006
start_eeg = 3041;
door_closed = 6947;
start_flash = 12159;
gp_x = [300:500];
gp_y = [340:420];

%%%for 005 (video 0081)%%%
%%%lights are off at frame 3999
%%%mean value of black circle is about 52.2006
start_eeg = 3330;
door_closed = 12240;
start_flash = 3268+14400;
gp_x = [320:460];
gp_y = [330:400];

%%%for 006 (video 0082?)%%%
%%%lights are off at frame 5280
%%%mean value of black circle is about 52.2006
start_eeg = 567;
door_closed = 5040;
start_flash = 10446;
gp_x = [360:500];
gp_y = [340:385];

%%%for 007 (video 0083?)%%%
%%%lights are off at frame 5280
%%%mean value of black circle is about 52.2006
start_eeg = 1045;
door_closed = 7440;
start_flash = 12567;
off_flash = ;
gp_x = [360:480];
gp_y = [370:430];

%%%for 008 (video 0084?)%%%
%%%lights are off at frame 5280
%%%mean value of black circle is about 52.2006
start_eeg = 2053;
door_closed = 7680;
start_flash = 12673;
off_flash = ;
gp_x = [380:500];
gp_y = [385:420];

%%%for 009 (video 0084?)%%%
%%%lights are off at frame 5280
%%%mean value of black circle is about 52.2006
start_eeg = 616;
door_closed = 6000;
start_flash = 11040;
gp_x = [360:460];
gp_y = [365:400];

%%%for 010 (video 0087)%%%
%%%lights are off at frame 5280
%%%mean value of black circle is about 52.2006
start_eeg = 1443;
door_closed = 8640;
start_flash = 13343;
gp_x = [380:500];
gp_y = [385:420];

%%%for 011 (video 0088)%%%
%%%lights are off at frame 5280
%%%mean value of black circle is about 52.2006
start_eeg = 638;
door_closed = 8400;
start_flash = 13176;
gp_x = [390:520];
gp_y = [330:380];


test1_gp = [];

%%%need to load the videos in chunks, not enough memory to load all frames?
%%%pi records at 90fps max, Go Pro at 240fps max%%%
%%%1 minute equals 5400 frames for pi, 14400 for go pro%%%
%%%go pro video 0078 is 120fps at 1280x720%%%
%%%1 minute is 7200

gp_video = [];
gp_frame_start = 1;
gp_frame_end = 7200;
gp_frame_minute = 7200;

%%%first we will loop through our video%%%
%%%the loop below will load the video in segments%%%
%%%and only store specific pixels (corresponding to the location of %%%
%%%one LED) and storing this information in a variable.%%%
%%%this way we can easily analyse all the frames at once%%%

%%%now we will loop through our Go Pro video%%%
%%%may have to change the below value depending on the video and fps%%%
for i_loop = 1:ceil(gp_frames/gp_frame_minute)
    if i_loop > 1
        gp_frame_start = (gp_frame_minute*(i_loop-1))+1;
        
        if rem(gp_frames,gp_frame_minute) > 0 && i_loop == ceil(gp_frames/gp_frame_minute)
            gp_frame_end = gp_frames;
        else
            gp_frame_end = (gp_frame_minute*i_loop);
        end
    end
    
    test1_gp(1,i_loop) = gp_frame_start;
    test1_gp(2,i_loop) = gp_frame_end;
    
    gp_video_temp = read(video_gp_ref,[gp_frame_start,gp_frame_end]);
    
    gp_video = cat(4,gp_video,gp_video_temp(gp_y,gp_x,:,:));
    
    gp_video_temp = [];
end


%%%%%preview specific frame of video%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
imagesc(gp_video(:,:,:,11131));
find(mean(mean(mean(gp_video(:,:,[2:3],door_closed:end),1),2),3),1)
% %
% % %%%find start flash
% % find(mean(mean(mean(gp_video(:,:,[2:3],[9941:end]),1),2),3) > 1,1)
% %
% % %%%find stop flash
% % find(mean(mean(mean(gp_video(:,:,[2:3],[9936:end]),1),2),3) < 1,1)


%%%for green flashes, look for minimum green value of 8
%%%green flash at 9616
%%%for blue flashes, look for a minimum blue value of 7
%%%blue flash at 10455
%%%if g and b are averaged, green flashes are about 6 and blue is 3

%%%now we will loop through our truncated videos%%%
%%%to find the start of each flash%%%

start_flash_gp = start_flash;
% % start_flash_gp = find(mean(mean(mean(gp_video(:,:,:,:)))) > 1.0,1);
% % start_exp_gp = start_flash_gp;
flash_latencies_gp = [];
flash_latencies_gp(1) = start_flash;

trial_type = [];

%%%determine if our first flash is a target or standard%%%
%%%if the average green value of the LED is greater than the average blue
%%%value, flash is likely a standard%%%
%%%we will also add an extra frame to the one we are testing, since
%%%sometimes when the LED first turns on the green/blue values are more
%%%similar
if mean(mean(gp_video(:,:,2,start_flash+1),1),2) > mean(mean(gp_video(:,:,3,start_flash+1),1),2)
trial_type(1) = 1;
%%%of if the blue value is greater, flash is likely a target%%%
elseif mean(mean(gp_video(:,:,3,start_flash+1),1),2) > mean(mean(gp_video(:,:,2,start_flash+1),1),2)
    trial_type(1) = 2;
end

%%%now go through the rest of the flashes%%%
for i_flash = 2:150
    stop_flash_gp = find(mean(mean(mean(gp_video(:,:,[2:3],start_flash_gp+1:end),1),2),3) < 0.1,1)+start_flash_gp;
    start_flash_gp = find(mean(mean(mean(gp_video(:,:,[2:3],stop_flash_gp+1:end),1),2),3) > 0.1,1)+stop_flash_gp;
    
    %%%if there is more green than blue
    if mean(mean(gp_video(:,:,2,start_flash_gp+1),1),2) > mean(mean(gp_video(:,:,3,start_flash_gp+1),1),2)
        trial_type(i_flash) = 1;
    elseif mean(mean(gp_video(:,:,2,start_flash_gp+1),1),2) < mean(mean(gp_video(:,:,3,start_flash_gp+1),1),2)
        trial_type(i_flash) = 2;
    end
    
    flash_latencies_gp(i_flash) = start_flash_gp;
end

% %  Eden's Messing around in hopes of improving green/blue differentiation
%%%for the go pro%%%
for i_flash = 2:150
    stop_flash_gp = find(mean(mean(mean(gp_video(:,:,[2:3],start_flash_gp+1:end),1),2),3) < 0.1,1)+start_flash_gp;
    start_flash_gp = find(mean(mean(mean(gp_video(:,:,[2:3],stop_flash_gp+1:end),1),2),3) > 0.1,1)+stop_flash_gp;
    
    %%%if there is more green than blue
    if ((mean(mean(gp_video(:,:,2,start_flash_gp),1),2)+ mean(mean(gp_video(:,:,2,start_flash_gp+1),1),2)+ mean(mean(gp_video(:,:,2,start_flash_gp+2),1),2))/3) > ((mean(mean(gp_video(:,:,3,start_flash_gp),1),2)+ mean(mean(gp_video(:,:,3,start_flash_gp+1),1),2)+ mean(mean(gp_video(:,:,3,start_flash_gp+2),1),2))/3)
        trial_type(i_flash) = 1;
    elseif ((mean(mean(gp_video(:,:,3,start_flash_gp),1),2)+ mean(mean(gp_video(:,:,3,start_flash_gp+1),1),2)+ mean(mean(gp_video(:,:,3,start_flash_gp+2),1),2))/3) > ((mean(mean(gp_video(:,:,2,start_flash_gp),1),2)+ mean(mean(gp_video(:,:,2,start_flash_gp+1),1),2)+ mean(mean(gp_video(:,:,2,start_flash_gp+2),1),2))/3)
        trial_type(i_flash) = 2;
    end
    
    flash_latencies_gp(i_flash) = start_flash_gp;
end

%%%let's plot our flashes to make sure verything worked
figure;
for i_plot = 1:length(flash_latencies_gp)
    
    subplot(15,10,i_plot);imagesc(gp_video(:,:,:,flash_latencies_gp(i_plot)));
    
    if trial_type(i_plot) == 1
        title ("Green")
    elseif trial_type(i_plot) == 2
        title ("Blue")
    end
    
end

%%% Trial Count %%%
count1=0;
count2=0;
for i_type = 1:length(trial_type)
    if trial_type(i_type) == 1
        count1 = count1 + 1
    elseif trial_type(i_type) == 2
        count2 = count2 + 1;
    end
end

count1
count2

%%% Transform(Switch to seconds) - Trim(set with EEG) - Adjust(stretch
%%% factor)
flash_latencies_gp_nonshifted = flash_latencies_gp;
flash_latencies_gp_nonshifted = flash_latencies_gp_nonshifted/240;
flash_latencies_gp_shifted = flash_latencies_gp - start_eeg;
flash_latencies_gp_shifted = flash_latencies_gp_shifted/240;
flash_latencies_gp_adjusted_shifted = (flash_latencies_gp_shifted*1.001)+0.12043;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%FOR TESTING%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Now we will go through the EEG file and determine our latencies%%%
filepath = ['M:\Experiments\Visual P3\EEG_Data'];
% filename = ['003_camera_p3.vhdr'];
filename = ['009_camera_p3.vhdr'];
% filename = ['005_camera_p3.vhdr'];

[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
EEG = pop_loadbv(filepath, filename, [], []);
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'setname','timingtest','gui','off');

start_trigger = ALLEEG(1).event(3).latency;
EEG_latencies_nonshifted = zeros(1,(length(ALLEEG(1).event)-3));
EEG_latencies_nonshifted(1) = start_trigger/ALLEEG.srate;

for i_event = 4:length(ALLEEG(1).event)-1
    
    EEG_latencies_shifted(i_event-2) = (ALLEEG(1).event(i_event).latency - start_trigger)/ALLEEG.srate;
    EEG_latencies_nonshifted(i_event-2) = (ALLEEG(1).event(i_event).latency)/ALLEEG.srate;
    
end


%%%now let's load our saved latencies and compare them%%%
load('M:\Experiments\Visual P3\Times\GOPRO_Times_ERP_1.mat')
load('M:\Experiments\Visual P3\Times\EEG_Times_ERP_1.mat')

%%%now let's load the times recorded by the pi%%%
%%%need to subtract 5 from these since there is 5 seconds before
%%%the red LEDs, indicating the start, are turned on%%%
pi_recorded_times = csvread('M:\Experiments\Visual P3\EEG_Data\000_06_all_visual_p3_trigs_amp.csv',1,0,[1,0,1,249]);
pi_recorded_times = pi_recorded_times - 5;

all_latencies(1,:) = EEG_latencies_nonshifted;
all_latencies(2,:) = flash_latencies_gp_shifted;
all_latencies(3,:) = flash_latencies_gp_nonshifted;

%%%some stats%%%
mean(all_latencies(1,:) - all_latencies(4,:))
min(all_latencies(1,:) - all_latencies(4,:))
max(all_latencies(1,:) - all_latencies(4,:))

%%%now let's plot each of our latencies%%%
conditions = {'EEG','GO Pro Shifted','GO Pro Unshifted'};

figure;hold on;
colours = ['r','g','b'];

for i_plot = 1:3
    
    plot(all_latencies(i_plot,:),[1:150],'color',colours(i_plot));
    
end
xlabel('Time (Seconds)');ylabel('Trial');legend('EEG','GO Pro Shifted','GO Pro Unshifted');
hold off;

%%%now let's plot each of our difference latencies%%%
conditions = {'GO Pro Shifted','GO Pro Unshifted'};

figure;hold on;
colours = ['r','g','b'];

for i_plot = 2:3
    
    plot(all_latencies(1,:)-all_latencies(i_plot,:),[1:250],'color',colours(i_plot));
    
end
xlabel('Time (Seconds)');ylabel('Trial');legend('GO Pro Shifted','GO Pro Unshifted');
hold off;


%%%create histograms of the latency difference from the EEG%%%
figure;hold on;

for i_plot = 2:4
    
    %    subplot(4,1,i_plot);hist((all_latencies(1,:)-all_latencies(i_plot,:))-mean(all_latencies(1,:)-all_latencies(i_plot,:)),10);
    subplot(3,1,i_plot-1);hist(all_latencies(1,:)-all_latencies(i_plot,:),25);
    title(['EEG Latencies Minus ' conditions{i_plot} ' Latencies']);
    xlabel('Time (Seconds)');ylabel('Counts');
    
end

hold off;

%%%create histograms of the latency difference from the EEG%%%
figure;hold on;

for i_plot = 2:4
    
    hist(all_latencies(1,:)-all_latencies(i_plot,:),25);
    xlabel('Time (Seconds)');ylabel('Counts');
    xlim([-1,18]);
    
end

hold off;


%%%now let's test the photo sensor to the triggers%%%
filepath = ['M:\Experiments\Visual P3\EEG_Data'];
filename = ['led_trigger_test_3.vhdr'];

[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
EEG = pop_loadbv(filepath, filename, [], []);
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'setname','timingtest','gui','off');

led_latencies_1 = [];
led_latencies_2 = [];
led_latencies_3 = [];

for i_event = 2:(length(ALLEEG(1).event)-1)
    if i_event == 3
        led_latencies_1(i_event-1) = (find(ALLEEG(1).data(17,(ALLEEG(1).event(i_event).latency:end)) < -600,1));
    else
        led_latencies_1(i_event-1) = (find(ALLEEG(1).data(17,(ALLEEG(1).event(i_event).latency:end)) < -100,1));
    end
    led_latencies_2(i_event-1) = (find(ALLEEG(1).data(17,(ALLEEG(1).event(i_event).latency:end)) < -100,1) + ALLEEG(1).event(i_event).latency);
    led_latencies_3(i_event-1) = (ALLEEG(1).event(i_event).latency);
    
end


figure;hold on;
plot(1:length(ALLEEG.times),ALLEEG.data(17,:));

for i_line = 1:length(led_latencies_3)
    
    line([led_latencies_2(i_line),led_latencies_2(i_line)],[-5000,5000],'color','r');
    line([led_latencies_3(i_line),led_latencies_3(i_line)],[-5000,5000],'color','k');
    
end
hold off;

mean(led_latencies_1)/ALLEEG.srate
min(led_latencies_1)/ALLEEG.srate
max(led_latencies_1)/ALLEEG.srate
std(led_latencies_1)/ALLEEG.srate


for i_time=1:length(trig_times); line([trig_times(i_time) trig_times(i_time)],[0 1800],'color','b'); end;hold off;
title([num2str(exp.participants{i_part}) ' Plot of Triggers Obtained from Muse AUX Data']);


%%%make a scatter plot of our latencies%%%

figure;hold on;
scatter(all_latencies(1,:),all_latencies(1,:),'r');
scatter(all_latencies(2,:),all_latencies(1,:),'g');
scatter(all_latencies(3,:),all_latencies(1,:),'b');
line([0,400],[0,400],'color','k');
hold off;
ylabel(['EEG Times']); xlabel(['Camera Times']);legend({'EEG Unshifted','GoPro Shifted','GoPro Unshifted'},'location','southeast');

xlim([370,400]);ylim([370,400]);

%%%coeffiients for EEG data%%%
mdl = LinearModel.fit(all_latencies(1,:),all_latencies(1,:),'linear');
%%%slope = 1
%%%intercept = -1.3339e-13

%%%coeffiients for Go Pro Shifted data%%%
mdl = LinearModel.fit(all_latencies(2,:),all_latencies(1,:),'linear');
%%%%%old data%%%%%
%%%slope = 1.001
%%%intercept = 0.0023913
%%%%%new data%%%%%
%%%slope = 1.001
%%%intercept = 0.12043

%%%coeffiients for Go Pro Unshifted data%%%
mdl = LinearModel.fit(all_latencies(3,:),all_latencies(1,:),'linear');
%%%slope = 1.001
%%%intercept = -17.705


gp_cam_diff = all_latencies(1,:) - all_latencies(3,:);

x = all_latencies(2,:);
y = (all_latencies(2,:).*1.041)+0.04034;

%%%now let's adjust our data points to match the EEG data%%%
figure;hold on;
scatter(all_latencies(1,:),all_latencies(1,:),'r');
scatter((all_latencies(2,:)*1.001)+0.12043,all_latencies(1,:),'g');
scatter((all_latencies(3,:)*1.001)+0.12043,all_latencies(1,:),'b');
line([0,150],[0,150],'color','k');
hold off;
ylabel(['EEG Times']); xlabel(['Camera Times']);legend({'EEG Shifted','GoPro Shifted','GoPro Unshifted'},'location','southeast');

xlim([10,40]);ylim([10,40]);

adjusted_times_gopro = (all_latencies(2,:)*1.001)+0.12043;

figure;
subplot(2,1,1);hist(all_latencies(2,:)-all_latencies(1,:),25);
subplot(2,1,2);hist(((all_latencies(2,:)*1.001)+0.12043)-all_latencies(1,:),25);

