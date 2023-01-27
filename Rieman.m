%% Defalut Setting
clc;clear;

% period : from -1s to 5s with 0.2s shifting
t_len = 1.0:0.2:5.0;
% subject # x num of period x num of period
total_acc = zeros([28,length(t_len),length(t_len)]); 


%% Default Info Load
idx = zeros([10, 50]);
for i = 1:10
    idx(i,:) = randperm(50);
end

ii_list = 1:length(t_len); % index for splitting data
out_list = 1:10;
j_list = 1:length(t_len); % index for 
i_list = 1:50;

%% for parallel
delete(gcp('nocreate'))
parpool('local')

%% this loop for selected subject number(1 is subject 2)
% 
for num=[1:27]
    tic
    parfor (ii=ii_list)
        % parallel process with training dataset
        for jj = j_list
            % the shifting interval is 0.2 sec
            % basis is end time of period
            t2_train = 1+0.2*(ii-1);
            t2_test = 1+0.2*(jj-1);

            % start time
            t1_train = -t2_train + 3;
            t1_test = -t2_test +3;

            disp(['[TRAIN] mo period : ' , num2str(-t1_train), '~', num2str(t2_train)]) %화면 출력용임. train 데이터에 해당하는 disp
            disp(['[TEST] me period : ' , num2str(-t1_test), '~', num2str(t2_test)]) %화면 출력용임. test 데이터에 해당하는 disp

            [epoch_data_mo, epoch_label_mo] = RiemanFn(t1_train, t2_train, 113+(2*num), 114+(2*num));
            %             [epoch_data_mi, epoch_label_mi] = RiemanFn(t1, t2, 57+(2*num), 58+(2*num));
            [epoch_data_me, epoch_label_me] = RiemanFn(t1_test, t2_test, 1+(2*num), 2+(2*num));

            tenfold_accuracy = 0;
            left_data_mo = squeeze(epoch_data_mo( :, find(epoch_label_mo==1), :));
            right_data_mo = squeeze(epoch_data_mo( :, find(epoch_label_mo==2), :));
            %             left_data_mi = squeeze(epoch_data_mi( :, find(epoch_label_mi==1), :));
            %             right_data_mi = squeeze(epoch_data_mi( :, find(epoch_label_mi==2), :));
            left_data_me = squeeze(epoch_data_me( :, find(epoch_label_me==1), :));
            right_data_me = squeeze(epoch_data_me( :, find(epoch_label_me==2), :));

            % Setting accuracy list for 10-Fold Cross Validation
            total_accuracy_M = zeros([10, 1]);

            % 10-Fold Cross Validation
            for out= out_list
                index = idx(out, :);
                % Data Load from dataset folder
                mo_l = left_data_mo(:, index, :);   mo_r = right_data_mo(:, index, :); % ch X trial X timepoint
                %                 mi_l = left_data_mi(:, index, :);   mi_r = right_data_mi(:, index, :);
                me_l = left_data_me(:, index, :);   me_r = right_data_me(:, index, :);
                
                % Covariance with data
                % cov_mi_l = zeros([19, 19, 50]);
                cov_me_l = zeros([19, 19, 50]);
                cov_mo_l = zeros([19, 19, 50]);
                % cov_mi_r = zeros([19, 19, 50]);
                cov_me_r = zeros([19, 19, 50]);
                cov_mo_r = zeros([19, 19, 50]);

                for i = i_list
                    tmp = squeeze(mo_l(:, i, :));% ch X trial X timepoint
                    cov_mo_l(:, :, i) = (tmp*tmp')/trace(tmp*tmp');
                    tmp = squeeze(mo_r(:, i, :));% ch X trial X timepoint
                    cov_mo_r(:, :, i) = (tmp*tmp')/trace(tmp*tmp');

                    %                     tmp = squeeze(mi_l(:, i, :));% ch X trial X timepoint
                    %                     cov_mi_l(:, :, i) = (tmp*tmp')/trace(tmp*tmp');
                    %                     tmp = squeeze(mi_r(:, i, :));% ch X trial X timepoint
                    %                     cov_mi_r(:, :, i) = (tmp*tmp')/trace(tmp*tmp');

                    tmp = squeeze(me_l(:, i, :));% ch X trial X timepoint
                    cov_me_l(:, :, i) = (tmp*tmp')/trace(tmp*tmp');
                    tmp = squeeze(me_r(:, i, :));% ch X trial X timepoint
                    cov_me_r(:, :, i) = (tmp*tmp')/trace(tmp*tmp');
                end

                % setting for the Rieman-CSP
                m = struct();
                ind = [1:100];
                ind = ind(randperm(length(ind))); %random index
                train_data = cat(3, cov_mo_l, cov_mo_r);
                train_data = train_data(:,:,ind);
                train_label = [ones(size(cov_mo_l,3),1); ones(size(cov_mo_r,3),1)+1];
                train_label = train_label(ind);

                test_data = cat(3,cov_me_l,cov_me_r);
                test_label = [ones(size(cov_me_l,3),1); ones(size(cov_me_r,3),1)+1];


                m.data = zeros([19, 19, 100]);
                m.labels = zeros([1, 100]);
                m.data = cat(3,train_data,test_data); % cat left and right covariance => ch X ch X trial*2
                m.labels=[train_label; test_label]'; % 1 X trial*2

                m.idxTraining = [1:100];
                m.idxTest = [101:200];

                %% Intialize data
                initformat = get(0,'format');
                format('long');
                display = false;

                %% Put the set structure in the FgMDM
                [FgMDM_acc, FgMDM_dect, FgMDM_dist]= FgMDM(m, 'Supervised', display); % normal FgMDM
                total_accuracy_M(out) = FgMDM_acc; % sub X 10 X 5

            end
            
            %% Calculate the 10-Fold Accuracy by Averaging
            tenfold_accuracy = mean(total_accuracy_M, 'all'); % sub X 10 X 5
            tenfivefold_std_ME = std(total_accuracy_M,1,'all');
            total_acc(num, ii,jj) = tenfold_accuracy;
            disp(['acc(', num2str(tenfold_accuracy),'%) stored in [', num2str(ii), ',', num2str(jj),']th array']);
            disp('%%%%%%%%%%%%%%%%%%%%%%%%%%');
            disp(['10 by 10 fold accuracy ' num2str(tenfold_accuracy),'%']);
            disp('=====================================');
        end
    end

    % Store the 10-fold Accuracy in your Desktop
    filepath = '__put your path__'
    fileName = [filepath, '/ACC_', num2str(num+1), '.mat'];
    eval(['acc_', num2str(num+1), '=', 'squeeze(total_acc(', num2str(num), ', :, :));']);
    eval(['save(fileName, "acc_', num2str(num+1),'");']);
    toc
end
% toc
delete(gcp('nocreate'))





