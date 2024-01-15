function allright = QD_L2_run2 (cond1, cond2, subBLK)
% clear;
% close all;
% cond1=3;
% cond2=5;

NAMES = { 
    
'1463_Jun01';'1807_May02';'2429_Jun05';
'2658_May03';'2738_May01'; '638_May30'; 
'2748_Jun06';'2762_May17';'2775_Jun21';
'2825_Jul05'; '202_May31'; '248_May25';
 '250_May17'; '418_Jun21'; '419_May11';
 '546_Aug24'; '625_Jun20'; '637_May28';
'2739_May07';};

mask = load_img ('avg4_4mm_brainmask','l');
load subj_names.mat;
load motion_mask.mat;

brain_coords = reshape(find(mask),1,[]);
% new! mask out voxels susceptible to motion
brain_coords = setdiff (brain_coords, motion_mask);

newmask = zeros(size(mask)); newmask(brain_coords) = 1;

Range =   2*subBLK-1; % 21:5:96];
numBLK = 17;

Label_full = [zeros(1,8*numBLK) ones(1,8*numBLK)];
Label_splt = [zeros(1,  subBLK) ones(1,  subBLK)];
Label_held = [zeros(1,6*numBLK) ones(1,6*numBLK)];

% init before first subject
data_cl0 = zeros(length(brain_coords),numBLK,8);
data_cl1 = zeros(length(brain_coords),numBLK,8);

%%

for (subj_idx=1:18)
    
    [subj_idx subj_idx subj_idx],

    % load the data blocks
    load( strcat('matfiles_NU/prepdata_',NAMES{subj_idx},'.mat') );
    % -----------------------------------------------------
    % -----------------------------------------------------
    if(cond1==1)
        
        num11 = round(numBLK/2);
        num22 = numBLK-num11;
        
        for(k=1:8)  data_cl0(:,:,k) = [dataCond{cond1,(k-1)*4 + 1}(:,1:num11) dataCond{cond1,(k-1)*4 + 3}(:,1:num22)];
        end        
    else
        for(k=1:8)  data_cl0(:,:,k) = dataCond{cond1,k}(:,1:numBLK);
        end
    end
    % -----------------------------------------------------
    for    (k=1:8)  data_cl1(:,:,k) = dataCond{cond2,k}(:,1:numBLK);
    end
    % -----------------------------------------------------
    % -----------------------------------------------------

    % full data matrix -- remove adjusted mean!!!
    full_data = [reshape( data_cl0, length(brain_coords),[],1 ) reshape( data_cl1, length(brain_coords),[],1 )];
    full_data = full_data - repmat( mean(full_data,2), [1 size(full_data,2)] );
    regr      = median(full_data)';
    full_data = detrend_matrix( full_data,0,regr(:),Label_full(:) );
    % re-split by class:
    data_cl0 = reshape( full_data(:, Label_full==0),  length(brain_coords),size(data_cl0,2),size(data_cl0,3) );
    data_cl1 = reshape( full_data(:, Label_full==1),  length(brain_coords),size(data_cl1,2),size(data_cl1,3) );

    % Kernel on full data set (used for reference)
     all_K = full_data'*full_data;
    % >>> dims x lambda matrix of discriminants
    model_all = QD_L2_create (all_K, Label_full, Range);
    map_ref   = QD_map_signed_set (model_all, full_data, all_K, 'L2' );
        % flip to match the CV scores:
    CVscores = full_data' * map_ref; % cv scores {timepts x dims}
    sggn = sign( mean(CVscores( Label_full==1,: )) - mean(CVscores( Label_full==0,: )) ); % 
    % now flip-em to match
    map_ref = map_ref * diag(sggn);

%% LD-PC, 4x20 blocks (LVL 111)
    
    n_splits = 4;
    % initialize
    
    SPLITDX = [1 2, 3 4, 5 6, 7 8; 
               3 4, 1 2, 5 6, 7 8; 
               5 6, 1 2, 3 4, 7 8;         
               7 8, 1 2, 3 4, 5 6 ];
    
    for split = 1:n_splits

        disp(split);
        list = SPLITDX(split,:);
        
        % get the data splits (held)
        held_data0 = reshape( data_cl0(:,:,list(3:8)), length(brain_coords),[],1 );
        held_data1 = reshape( data_cl1(:,:,list(3:8)), length(brain_coords),[],1 );
        % concatenate
        held_data = [held_data0 held_data1];
        mean_held = mean(held_data,2);
        %
        held_data = held_data  - repmat(mean_held,[1 size(held_data ,2)]); 
        
        PP   = zeros( Range, 2 );
        RR   = zeros( Range, 2 );
        GG   = zeros( Range, 2 );
        rSPM = zeros( length(brain_coords), Range, 2 );
        
        for(swap=1:2)
        
            if(swap==1)
                % get the data splits (train)
                trn_data0 = reshape( data_cl0(:,:,list(1)), length(brain_coords),[],1 );
                trn_data1 = reshape( data_cl1(:,:,list(1)), length(brain_coords),[],1 );
                % get the data splits (test)
                tst_data0 = reshape( data_cl0(:,:,list(2)), length(brain_coords),[],1 );
                tst_data1 = reshape( data_cl1(:,:,list(2)), length(brain_coords),[],1 );
            else
                % get the data splits (train)
                trn_data0 = reshape( data_cl0(:,:,list(1)), length(brain_coords),[],1 );
                trn_data1 = reshape( data_cl1(:,:,list(2)), length(brain_coords),[],1 );
                % get the data splits (test)
                tst_data0 = reshape( data_cl0(:,:,list(2)), length(brain_coords),[],1 );
                tst_data1 = reshape( data_cl1(:,:,list(1)), length(brain_coords),[],1 );
            end            
            
            % concatenate
            trn_data = [trn_data0(:,end-subBLK+1:end) trn_data1(:,end-subBLK+1:end)];
            tst_data = [tst_data0(:,end-subBLK+1:end) tst_data1(:,end-subBLK+1:end)];

            % == mean-centering == %
            mean_trn = mean(trn_data,2);
            mean_tst = mean(tst_data,2);
            %
            trn_data  = trn_data  - repmat(mean_trn,[1 size(trn_data ,2)]);
            trn_data0 = trn_data0 - repmat(mean_trn,[1 size(trn_data0,2)]);
            trn_data1 = trn_data1 - repmat(mean_trn,[1 size(trn_data1,2)]);
            %
            tst_data  = tst_data  - repmat(mean_tst,[1 size(tst_data ,2)]); 
            tst_data0 = tst_data0 - repmat(mean_tst,[1 size(tst_data0,2)]);
            tst_data1 = tst_data1 - repmat(mean_tst,[1 size(tst_data1,2)]);
            % == mean-centering == %

            % calculate class means (in voxel space)
            trn_avg.mean0 = mean (trn_data0, 2);
            trn_avg.mean1 = mean (trn_data1, 2);
            tst_avg.mean0 = mean (tst_data0, 2);
            tst_avg.mean1 = mean (tst_data1, 2);

            % training/test kernels
             trn_K = trn_data'*trn_data;
             tst_K = tst_data'*tst_data;

            model_trn = QD_L2_create (trn_K, Label_splt, Range);
            map_trn   = QD_map_signed_set (model_trn, trn_data, trn_K, 'L2' );
            %
            CC=corrcoef([map_ref ,map_trn]);
            CC=sign(diag( CC(1:Range,Range+1:end) ));
            map_trn=map_trn*diag(CC);

            model_tst = QD_L2_create (tst_K, Label_splt, Range);
            map_tst   = QD_map_signed_set (model_tst, tst_data, tst_K, 'L2' );
            %
            CC=corrcoef([map_ref ,map_tst]);
            CC=sign(diag( CC(1:Range,Range+1:end) ));
            map_tst=map_tst*diag(CC);

            for(r=1:Range)
                [RR(r,swap) spm] = get_rSPM( map_trn(:,r), map_tst(:,r), 1 );
                % add it in:
                 rSPM(:,r,swap) = rSPM(:,r) + spm;
            end

            % classify test data using training model
            [accur_1] = QD_classify_set (model_trn, trn_data, trn_avg, tst_data, Label_splt,'L2');        
            % classify training data using test model
            [accur_2] = QD_classify_set (model_tst, tst_data, tst_avg, trn_data, Label_splt,'L2');        
            
            PP(:,swap) = (accur_1 + accur_2) ./ 2;

            %% %% %% -------- %% %% %% -------- %% %% %%

                % classify test data using training model
                [genr_1] = QD_classify_set (model_trn, trn_data, trn_avg, held_data, Label_held,'L2');        
                % classify training data using test model
                [genr_2] = QD_classify_set (model_tst, tst_data, tst_avg, held_data, Label_held,'L2');        

                
                GG(:,swap) = (genr_1 + genr_2) ./ 2;

            %% %% %% -------- %% %% %% -------- %% %% %%
        end
            
        RR_sub = mean(RR,2);
        PP_sub = mean(PP,2);
        DD_sub = sqrt( (1-RR_sub).^2 + (1-PP_sub).^2 );
        RPD_sub = [RR_sub PP_sub DD_sub];
        GG_sub = mean(GG,2);
        rSPM_sub = mean(rSPM,3);
        
        save(strcat('QD_L2_sub',num2str(subBLK),'alt_C',num2str(10*cond1+cond2),'_subj_',num2str(subj_idx),'_',num2str(split),'.mat'),...
            'rSPM_sub','RPD_sub','GG_sub');
        
    end % of the splits loop
end

allright=1;