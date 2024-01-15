function [ output ] = LD_prepare_AND_classify_PP( data_trn, data_tst, REF_IMG, labels, data_HELD, labels_HELD, Range, RangeVal2, regtype)
% *corrected
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

rSPM = zeros( Nvox, Range );
RR   = zeros( Range, 1    );
PP   = zeros( Range, 1 );
GG   = zeros( Range, 1 );

if( strcmp(regtype,'L1') )

    % run LD
    model_trn = LD_L1_train(dataCov_trn, labels, Range);
    map_trn   = LD_map (model_trn, data_trn);
    %
    CC=corrcoef([REF_IMG ,map_trn]);
    CC=sign(diag( CC(1:Range,Range+1:end) ));
    map_trn=map_trn*diag(CC);
    
    % run LD
    model_tst = LD_L1_train(dataCov_tst, labels, Range);
    map_tst   = LD_map (model_tst, data_tst);
    %
    CC=corrcoef([REF_IMG ,map_tst]);
    CC=sign(diag( CC(1:Range,Range+1:end) ));
    map_tst=map_tst*diag(CC);
    
    % classify
    [accur_1] = LD_classify_PP (model_trn, data_trn, trn_avg, data_tst, labels);
    [accur_2] = LD_classify_PP (model_tst, data_tst, tst_avg, data_trn, labels);
    % prediction
    PP = (accur_1 + accur_2)./2;

    % classify
    [accur_1] = LD_classify_PP (model_trn, data_trn, trn_avg, data_HELD, labels_HELD);
    [accur_2] = LD_classify_PP (model_tst, data_tst, tst_avg, data_HELD, labels_HELD);
    % prediction
    GG = (accur_1 + accur_2)./2;
    
elseif( strcmp(regtype,'L2') )

    % run LD
    model_trn = LD_L2_train(dataCov_trn, labels, Range);
    map_trn   = LD_map (model_trn, data_trn);
    %
    CC=corrcoef([REF_IMG ,map_trn]);
    CC=sign(diag( CC(1:Range,Range+1:end) ));
    map_trn=map_trn*diag(CC);
    
    % run LD
    model_tst = LD_L2_train(dataCov_tst, labels, Range);
    map_tst   = LD_map (model_tst, data_tst);
    %
    CC=corrcoef([REF_IMG ,map_tst]);
    CC=sign(diag( CC(1:Range,Range+1:end) ));
    map_tst=map_tst*diag(CC);
    
    % classify
    [accur_1] = LD_classify_PP (model_trn, data_trn, trn_avg, data_tst, labels);
    [accur_2] = LD_classify_PP (model_tst, data_tst, tst_avg, data_trn, labels);
    % prediction
    PP = (accur_1 + accur_2)./2;

    % classify
    [accur_1] = LD_classify_PP (model_trn, data_trn, trn_avg, data_HELD, labels_HELD);
    [accur_2] = LD_classify_PP (model_tst, data_tst, tst_avg, data_HELD, labels_HELD);
    % prediction
    GG = (accur_1 + accur_2)./2;
    
elseif( strcmp(regtype,'PC') )
    
    % SVD on full data set (used for reference)
    [v, s] = svd (dataCov_trn); s=sqrt(s); trn_u=data_trn*v*inv(s);
     trn_Z = s*v';
    % SVD on full data set (used for reference)
    [v, s] = svd (dataCov_tst); s=sqrt(s); tst_u=data_tst*v*inv(s);
     tst_Z = s*v';

    % run LD
    model_trn = LD_NR_train(trn_Z, labels, Range);
    map_trn   = LD_map (model_trn, trn_u(:,1:Range));
    %
    CC=corrcoef([REF_IMG ,map_trn]);
    CC=sign(diag( CC(1:Range,Range+1:end) ));
    map_trn=map_trn*diag(CC);
    
    % run LD
    model_tst = LD_NR_train(tst_Z, labels, Range);
    map_tst   = LD_map (model_tst, tst_u(:,1:Range));
    %
    CC=corrcoef([REF_IMG ,map_tst]);
    CC=sign(diag( CC(1:Range,Range+1:end) ));
    map_tst=map_tst*diag(CC);

    % classify 
    [accur_1] = LD_classify_PP (model_trn, trn_u(:,1:Range), trn_avg, data_tst, labels);
    [accur_2] = LD_classify_PP (model_tst, tst_u(:,1:Range), tst_avg, data_trn, labels);
    %
    PP = (accur_1 + accur_2)./2;
    
    % classify 
    [accur_1] = LD_classify_PP (model_trn, trn_u(:,1:Range), trn_avg, data_HELD, labels_HELD);
    [accur_2] = LD_classify_PP (model_tst, tst_u(:,1:Range), tst_avg, data_HELD, labels_HELD);
    %
    GG = (accur_1 + accur_2)./2;
    
elseif(  strcmp(regtype,'IC') )
        
    % SVD on full data set (used for reference)
    [v, s] = svd (dataCov_trn); s=sqrt(s); trn_u=data_trn*v*inv(s);
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
    [v, s] = svd (dataCov_tst); s=sqrt(s); tst_u=data_tst*v*inv(s);
     tst_Z = s*v';
    % trim dimensionality
     tst_Z = tst_Z(1:RangeVal2,:);
     tst_u = tst_u(:,1:RangeVal2);

    %
    [tst_S A W] = fastica( tst_Z , 'verbose', 'off','g','tanh', 'numOfIC', RangeVal2 ); 
    lack = RangeVal2 - size(W,1);
    if( lack > 0 ) W = [W; rand( lack,  RangeVal2)]; end
    %
    tst_u_ic = tst_u*W';
    tst_u_ic = tst_u_ic./repmat( sqrt(sum(tst_u_ic.^2)), [size(tst_u_ic,1),1]);
    tst_S    = tst_u_ic'*data_tst;
     
    % run LD
    model_trn = LD_NR_train(trn_S, labels, Range);
    map_trn   = LD_map (model_trn, trn_u_ic(:,1:Range));
    %
    CC=corrcoef([REF_IMG ,map_trn]);
    CC=sign(diag( CC(1:Range,Range+1:end) ));
    map_trn=map_trn*diag(CC);
    
    % run LD
    model_tst = LD_NR_train(tst_S, labels, Range);
    map_tst   = LD_map (model_tst, tst_u_ic(:,1:Range));
    %
    CC=corrcoef([REF_IMG ,map_tst]);
    CC=sign(diag( CC(1:Range,Range+1:end) ));
    map_tst=map_tst*diag(CC);

    % classify 
    [accur_1] = LD_classify_PP (model_trn, trn_u_ic(:,1:Range), trn_avg, data_tst, labels);
    [accur_2] = LD_classify_PP (model_tst, tst_u_ic(:,1:Range), tst_avg, data_trn, labels);
    %
    PP = (accur_1 + accur_2)./2;  
    
    % classify 
    [accur_1] = LD_classify_PP (model_trn, trn_u_ic(:,1:Range), trn_avg, data_HELD, labels_HELD);
    [accur_2] = LD_classify_PP (model_tst, tst_u_ic(:,1:Range), tst_avg, data_HELD, labels_HELD);
    %
    GG = (accur_1 + accur_2)./2;       
end

%% ===================================================================== %%
%%                            SUMMARIZE DATA                             %%
%% ===================================================================== %%


% reproducibility estimation
for(r=1:Range)
    [RR(r) rSPM(:,r)] = get_rSPM( map_trn(:,r), map_tst(:,r), 1 );
end

DD = sqrt( (1-RR).^2 + (1-PP).^2 );

output.RR   = RR;
output.PP   = PP;
output.GG   = GG;
output.DD   = DD;
output.rSPM = rSPM;
