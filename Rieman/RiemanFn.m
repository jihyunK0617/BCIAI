function [epoch_data, epoch_label] = RiemanFn(t1, t2, f1, f2)
    num_subject = 28;
    time = 3;
    ch = 19; n_trials = 50;
    bp_low = 4; bp_high = 50;
    fpath = '/Users/hwangminjoo/Documents/MATLAB/BCILAB/datasets/matfile/all/';
    file_list = dir([fpath 'M*.mat']);
    num_interval = 41;
    epoch_data = zeros(ch,n_trials,300*time);
    epoch_label = zeros(n_trials);
    load('10_index.mat');

    for fileNum =  f1:f2%:length(file_list)
        load([fpath file_list(fileNum).name]); % load file
        disp(file_list(fileNum).name)
    
        if mod(fileNum,2) == 1
            subNum = (fileNum+1)/2;   
            session = 0;
        end
        
        % Trigger setting and Epoching from start to end trigger
        trigger = [find(event); event(find(event))]';
        start_time = trigger(find(trigger(:,2)==8),1);
        if  isempty(find(trigger(:,2)==9))
            end_time = length(event);
        else
            end_time = trigger(find(trigger(:,2)==9),1);
        end
    
        trigger = [find(event); event(find(event))]';
        cue_trigger = trigger(find(trigger(:,2) == 1 |trigger(:,2) == 2),1);
        cue_label = trigger(find(trigger(:,2) == 1 |trigger(:,2) == 2),2);
    
        % Channel selection
        m_data = data([1:8 11:16 19:20 23:24 27], :); % 필요한 채널의 데이터만 가져옴 
        chanlocs = chanlocs([1:8 11:16 19:20 23:24 27]);
    
        % CAR
        car_signal = m_data - repmat(mean(m_data,1),19,1);
    
        % Bandpass filtering
        bp_signal =  ft_preproc_bandpassfilter(car_signal, srate,[4 50], 4,'but');
    
        for epoch = 1:length(cue_trigger)
            tmp_signal =  bp_signal(:,cue_trigger(epoch)-srate*t1+1:cue_trigger(epoch)+srate*t2);
            epoch_data(:,session*n_trials+epoch,:) = tmp_signal;
            epoch_label(session*n_trials+epoch) = cue_label(epoch);
        end
        i = 0;

        %%
        if session == 1
            return
        end
        session = 1;
    end
end