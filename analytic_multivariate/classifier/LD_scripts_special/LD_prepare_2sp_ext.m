function [ output ] = LD_prepare_2sp( data_trn, data_tst, REF_IMG, labels, Range, Curt, regtype, mask )
%
% .For now, assume DATA_cl0/DATA_cl1 are equal sized (3d matrixes)
%

% matrix dimensions
[Nvox Nsamp] = size( data_trn );

% full data matrix, mean centered -- then covariance!
data_trn    = data_trn - repmat(mean(data_trn,2),[1 Nsamp]);
dataCov_trn = data_trn'*data_trn;
% --
data_tst    = data_tst - repmat(mean(data_tst,2),[1 Nsamp]);
dataCov_tst = data_tst'*data_tst;
% calculate class means (in voxel space)
trn_avg.mean0 = mean (data_trn(:,labels==0), 2);
trn_avg.mean1 = mean (data_trn(:,labels==1), 2);
tst_avg.mean0 = mean (data_tst(:,labels==0), 2);
tst_avg.mean1 = mean (data_tst(:,labels==1), 2);

%% ===================================================================== %%
%%                              TRAIN DATA                               %%
%% ===================================================================== %%

rSPM = zeros( Nvox, Range );
PP   = zeros( Range, 1 );
RR   = zeros( Range, 1 );

if( strcmp(regtype,'PCL1') )

    % SVD on full data set (used for reference)
    [trn_u, s, v] = svd (dataCov_trn); s=sqrt(s); trn_u=data_trn*v*inv(s);
     trn_Z = s*v';
    % SVD on full data set (used for reference)
    [tst_u, s, v] = svd (dataCov_tst); s=sqrt(s); tst_u=data_tst*v*inv(s);
     tst_Z = s*v';

     trn_Z = trn_Z(1:Curt,:); trn_u=trn_u(:,1:Curt);
     tst_Z = tst_Z(1:Curt,:); tst_u=tst_u(:,1:Curt);
     
    % run LD
    model_trn = LD_L1_train(trn_Z, labels, Range);
    map_trn   = LD_map (model_trn, trn_u);
    %
    CC=corrcoef([REF_IMG ,map_trn]);
    CC=sign(diag( CC(1:Range,Range+1:end) ));
    map_trn=map_trn*diag(CC);

    % run LD
    model_tst = LD_L1_train(tst_Z, labels, Range);
    map_tst   = LD_map (model_tst, tst_u);
    %
    CC=corrcoef([REF_IMG ,map_tst]);
    CC=sign(diag( CC(1:Range,Range+1:end) ));
    map_tst=map_tst*diag(CC);

    % classify
    [accur_1] = LD_classify_set (model_trn, trn_u, trn_avg, data_tst, labels);
    [accur_2] = LD_classify_set (model_tst, tst_u, tst_avg, data_trn, labels);
    % prediction
    PP = (accur_1 + accur_2)./2;

elseif( strcmp(regtype,'PCL2') )

    % SVD on full data set (used for reference)
    [trn_u, s, v] = svd (dataCov_trn); s=sqrt(s); trn_u=data_trn*v*inv(s);
     trn_Z = s*v';
    % SVD on full data set (used for reference)
    [tst_u, s, v] = svd (dataCov_tst); s=sqrt(s); tst_u=data_tst*v*inv(s);
     tst_Z = s*v';

     trn_Z = trn_Z(1:Curt,:); trn_u=trn_u(:,1:Curt);
     tst_Z = tst_Z(1:Curt,:); tst_u=tst_u(:,1:Curt);
     
    % run LD
    model_trn = LD_L2_train(trn_Z, labels, Range);
    map_trn   = LD_map (model_trn, trn_u);
    %
    CC=corrcoef([REF_IMG ,map_trn]);
    CC=sign(diag( CC(1:Range,Range+1:end) ));
    map_trn=map_trn*diag(CC);

    % run LD
    model_tst = LD_L2_train(tst_Z, labels, Range);
    map_tst   = LD_map (model_tst, tst_u);
    %
    CC=corrcoef([REF_IMG ,map_tst]);
    CC=sign(diag( CC(1:Range,Range+1:end) ));
    map_tst=map_tst*diag(CC);

    % classify
    [accur_1] = LD_classify_set (model_trn, trn_u, trn_avg, data_tst, labels);
    [accur_2] = LD_classify_set (model_tst, tst_u, tst_avg, data_trn, labels);
    % prediction
    PP = (accur_1 + accur_2)./2;

elseif( strcmp(regtype,'EN') )
    
    % run LD
    model_trn = LD_EN_train(dataCov_trn, labels, Curt, Range);
    map_trn   = LD_map (model_trn, data_trn);
    %
    CC=corrcoef([REF_IMG ,map_trn]);
    CC=sign(diag( CC(1:Range,Range+1:end) ));
    map_trn=map_trn*diag(CC);
    
    % run LD
    model_tst = LD_EN_train(dataCov_tst, labels, Curt, Range);
    map_tst   = LD_map (model_tst, data_tst);
    %
    CC=corrcoef([REF_IMG ,map_tst]);
    CC=sign(diag( CC(1:Range,Range+1:end) ));
    map_tst=map_tst*diag(CC);
    
    % classify
    [accur_1] = LD_classify_set (model_trn, data_trn, trn_avg, data_tst, labels);
    [accur_2] = LD_classify_set (model_tst, data_tst, tst_avg, data_trn, labels);
    % prediction
    PP = (accur_1 + accur_2)./2;
    
end

%% ===================================================================== %%
%%                            SUMMARIZE DATA                             %%
%% ===================================================================== %%

rSPM = zeros(Nvox, Range);
RR   = zeros(Range, 1 );

% reproducibility estimation
for(r=1:Range)
    [RR(r) rSPM(:,r)] = get_rSPM( map_trn(:,r), map_tst(:,r), 1 );
end

DD = sqrt( (1-RR).^2 + (1-PP).^2 );

output.RR   = RR;
output.PP   = PP;
output.DD   = DD;
output.rSPM = rSPM;
