function [ GG ] = LR_classifyOnly_2sp( data_trn, data_tst, labels, data_HELD, labels_HELD, Range, RangeVal2, regtype )
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

data_HELD    = data_HELD - repmat(mean(data_HELD,2),[1 size(data_HELD,2)]);

%% ===================================================================== %%
%%                              TRAIN DATA                               %%
%% ===================================================================== %%

GG   = zeros( Range, 1 );

if( strcmp(regtype,'L1') )

    % run LR
    model_trn = LR_L1_train(dataCov_trn, labels, Range);
    model_tst = LR_L1_train(dataCov_tst, labels, Range);
    
    % classify
    [accur_1] = LR_classify_set (model_trn, data_trn, trn_avg, data_HELD, labels_HELD);
    [accur_2] = LR_classify_set (model_tst, data_tst, tst_avg, data_HELD, labels_HELD);
    % prediction
    GG = (accur_1 + accur_2)./2;
    
elseif( strcmp(regtype,'L2') )
    
    % run LR
    model_trn = LR_L2_train(dataCov_trn, labels, Range);
    model_tst = LR_L2_train(dataCov_tst, labels, Range);
    
    % classify
    [accur_1] = LR_classify_set (model_trn, data_trn, trn_avg, data_HELD, labels_HELD);
    [accur_2] = LR_classify_set (model_tst, data_tst, tst_avg, data_HELD, labels_HELD);
    % prediction
    GG = (accur_1 + accur_2)./2;
    
elseif( strcmp(regtype,'PC') )
    
    % SVD on full data set (used for reference)
    [trn_u, s, v] = svd (dataCov_trn); s=sqrt(s); trn_u=data_trn*v*inv(s);
     trn_Z = s*v';
    % SVD on full data set (used for reference)
    [tst_u, s, v] = svd (dataCov_tst); s=sqrt(s); tst_u=data_tst*v*inv(s);
     tst_Z = s*v';

    % run LR
    model_trn = LR_NR_train(trn_Z, labels, Range);
    model_tst = LR_NR_train(tst_Z, labels, Range);

    % classify 
    [accur_1] = LR_classify_set (model_trn, trn_u(:,1:Range), trn_avg, data_HELD, labels_HELD);
    [accur_2] = LR_classify_set (model_tst, tst_u(:,1:Range), tst_avg, data_HELD, labels_HELD);
    %
    GG = (accur_1 + accur_2)./2;
    
elseif(  strcmp(regtype,'IC') )
        
    % SVD on full data set (used for reference)
    [trn_u, s, v] = svd (dataCov_trn); s=sqrt(s); trn_u=data_trn*v*inv(s);
     trn_Z = s*v';
    % trim dimensionality
     trn_Z = trn_Z(1:RangeVal2,:);
     trn_u = trn_u(:,1:RangeVal2);
    %

    [trn_S A W] = fastica( trn_Z , 'verbose', 'off','g','tanh', 'numOfIC', RangeVal2 ); 
    lack = RangeVal2 - size(W,1);
    if( lack > 0 ) W = [W; rand( lack,  RangeVal2)]; end
    %        
    trn_u_ic = trn_u*W';
    trn_u_ic = trn_u_ic./repmat( sqrt(sum(trn_u_ic.^2)), [size(trn_u_ic,1),1]);
    trn_S    = trn_u_ic'*data_trn;
     
    % SVD on full data set (used for reference)
    [tst_u, s, v] = svd (dataCov_tst); s=sqrt(s); tst_u=data_tst*v*inv(s);
     tst_Z = s*v';
    %
     tst_Z = tst_Z(1:RangeVal2,:);
     tst_u = tst_u(:,1:RangeVal2);


    [tst_S A W] = fastica( tst_Z , 'verbose', 'off','g','tanh', 'numOfIC', RangeVal2 ); 
    lack = RangeVal2 - size(W,1);
    if( lack > 0 ) W = [W; rand( lack,  RangeVal2)]; end
    %        
    tst_u_ic = tst_u*W';
    tst_u_ic = tst_u_ic./repmat( sqrt(sum(tst_u_ic.^2)), [size(tst_u_ic,1),1]);
    tst_S    = tst_u_ic'*data_tst;
     
    % run LR
    model_trn = LR_NR_train(trn_S, labels, Range);
    model_tst = LR_NR_train(tst_S, labels, Range);

    % classify 
    [accur_1] = LR_classify_set (model_trn, trn_u_ic(:,1:Range), trn_avg, data_HELD, labels_HELD);
    [accur_2] = LR_classify_set (model_tst, tst_u_ic(:,1:Range), tst_avg, data_HELD, labels_HELD);
    %
    GG = (accur_1 + accur_2)./2;
    
end

%% ===================================================================== %%
%%                            SUMMARIZE DATA                             %%
%% ===================================================================== %%
