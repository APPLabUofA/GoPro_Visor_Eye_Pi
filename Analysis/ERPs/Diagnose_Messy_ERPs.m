% Pulls originial bvmknum latency from ALLEEG struct
for i_EEG = 1:length(ALLEEG)
    count = 1;
    for i_event = 1:length([ALLEEG(i_EEG).urevent(:).latency])
        if count <= length([ALLEEG(i_EEG).event(:).bvmknum])
            if ALLEEG(i_EEG).urevent(i_event).bvmknum == ALLEEG(i_EEG).event(count).bvmknum
                [ALLEEG(i_EEG).event(count).start_latency] = [ALLEEG(i_EEG).urevent(i_event).latency];
                count = count + 1;  
            end
        end
    end
end