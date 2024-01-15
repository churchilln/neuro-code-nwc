function [ output ] = QD_prepare_AND_classify( data_trn, data_tst, REF_IMG, labels, data_HELD, labels_HELD, Range, RangeVal2, regtype, mask )
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

% % CANNOT PREDEFINE DUE TO L1:
% % rSPM = zeros( Nvox, Range );
% % RR   = zeros( Range, 1    );
% % PP   = zeros( Range, 1 );

if( strcmp(regtype,'L1') )

    FirstLim = Range+1;
    
     %% truncated range for GLASSO %%
    RangeGLASSO = min( [5 Range] );
    
    % predefiners
    rSPM = zeros( Nvox, RangeGLASSO );
    RR   = zeros( RangeGLASSO, 1    );
    PP   = zeros( RangeGLASSO, 1 );
    GG   = zeros( RangeGLASSO, 1 );
    
    % run LR
    model_trn = QD_L1_train (dataCov_trn, labels, FirstLim, Range);
    map_trn   = QD_map_signed_set (model_trn, data_trn, dataCov_trn, 'L1' );
    %
    CC=corrcoef([REF_IMG ,map_trn]);
    CC=sign(diag( CC(1:RangeGLASSO,RangeGLASSO+1:end) ));
    map_trn=map_trn*diag(CC);
    
    % run LR
    model_tst = QD_L1_train (dataCov_tst, labels, FirstLim, Range);
    map_tst   = QD_map_signed_set (model_tst, data_tst, dataCov_tst, 'L1' );
    %
    CC=corrcoef([REF_IMG ,map_tst]);
    CC=sign(diag( CC(1:RangeGLASSO,RangeGLASSO+1:end) ));
    map_tst=map_tst*diag(CC);
    
    % classify test data using training model
    [accur_1] = QD_classify_set (model_trn, data_trn, trn_avg, data_tst, labels,'L1');        
    [accur_2] = QD_classify_set (model_tst, data_tst, tst_avg, data_trn, labels,'L1');        
    % prediction
    PP = (accur_1 + accur_2)./2;
    
    % classify test data using training model
    [accur_1] = QD_classify_set (model_trn, data_trn, trn_avg, data_HELD, labels_HELD,'L1');        
    [accur_2] = QD_classify_set (model_tst, data_tst, tst_avg, data_HELD, labels_HELD,'L1');        
    % prediction
    GG = (accur_1 + accur_2)./2;

elseif( strcmp(regtype,'L2') )
    
    rSPM = zeros( Nvox, Range );
    RR   = zeros( Range, 1    );
    PP   = zeros( Range, 1 );
    GG   = zeros( Range, 1 );
    
    % run LR
    model_trn = QD_L2_train (dataCov_trn, labels, Range);
    map_trn   = QD_map_signed_set (model_trn, data_trn, dataCov_trn, 'L2' );
    %
    CC=corrcoef([REF_IMG ,map_trn]);
    CC=sign(diag( CC(1:Range,Range+1:end) ));
    map_trn=map_trn*diag(CC);
    
    % run LR
    model_tst = QD_L2_train (dataCov_tst, labels, Range);
    map_tst   = QD_map_signed_set (model_tst, data_tst, dataCov_tst, 'L2' );
    %
    CC=corrcoef([REF_IMG ,map_tst]);
    CC=sign(diag( CC(1:Range,Range+1:end) ));
    map_tst=map_tst*diag(CC);
    
    % classify test data using training model
    [accur_1] = QD_classify_set (model_trn, data_trn, trn_avg, data_tst, labels,'L2');        
    [accur_2] = QD_classify_set (model_tst, data_tst, tst_avg, data_trn, labels,'L2');        
    % prediction
    PP = (accur_1 + accur_2)./2;
    
    % classify test data using training model
    [accur_1] = QD_classify_set (model_trn, data_trn, trn_avg, data_HELD, labels_HELD,'L2');        
    [accur_2] = QD_classify_set (model_tst, data_tst, tst_avg, data_HELD, labels_HELD,'L2');        
    % prediction
    GG = (accur_1 + accur_2)./2;

elseif( strcmp(regtype,'PC') )
    
    rSPM = zeros( Nvox, Range );
    RR   = zeros( Range, 1    );
    PP   = zeros( Range, 1 );
    GG   = zeros( Range, 1 );
    
    % SVD on full data set (used for reference)
    [trn_u, s, v] = svd (dataCov_trn); s=sqrt(s); trn_u=data_trn*v*inv(s);
     trn_Z = s*v';
    % SVD on full data set (used for reference)
    [tst_u, s, v] = svd (dataCov_tst); s=sqrt(s); tst_u=data_tst*v*inv(s);
     tst_Z = s*v';

    % run LR
    model_trn = QD_NR_train (trn_Z, labels, Range);
    map_trn   = QD_map_signed_set (model_trn, trn_u(:,1:Range), trn_Z(1:Range,:),'NR' );
    %
    CC=corrcoef([REF_IMG ,map_trn]);
    CC=sign(diag( CC(1:Range,Range+1:end) ));
    map_trn=map_trn*diag(CC);
    
    % run LR
    model_tst = QD_NR_train (tst_Z, labels, Range);
    map_tst   = QD_map_signed_set (model_tst, tst_u(:,1:Range), tst_Z(1:Range,:),'NR' );
    %
    CC=corrcoef([REF_IMG ,map_tst]);
    CC=sign(diag( CC(1:Range,Range+1:end) ));
    map_tst=map_tst*diag(CC);

    % classify test data using training model
    [accur_1] = QD_classify_set (model_trn, trn_u(:,1:Range), trn_avg, data_tst, labels,'NR');        
    [accur_2] = QD_classify_set (model_tst, tst_u(:,1:Range), tst_avg, data_trn, labels,'NR');        
    %
    PP = (accur_1 + accur_2)./2;
    
    % classify test data using training model
    [accur_1] = QD_classify_set (model_trn, trn_u(:,1:Range), trn_avg, data_HELD, labels_HELD,'NR');        
    [accur_2] = QD_classify_set (model_tst, tst_u(:,1:Range), tst_avg, data_HELD, labels_HELD,'NR');        
    %
    GG = (accur_1 + accur_2)./2;

elseif(  strcmp(regtype,'IC') )
        
    rSPM = zeros( Nvox, Range );
    RR   = zeros( Range, 1    );
    PP   = zeros( Range, 1 );
    GG   = zeros( Range, 1 );
    
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
    [tst_S A W] = fastica( tst_Z , 'verbose', 'off','g','tanh', 'numOfIC', RangeVal2 ); 
    lack = RangeVal2 - size(W,1);
    if( lack > 0 ) W = [W; rand( lack,  RangeVal2)]; end
    %
    tst_u_ic = tst_u*W';
    tst_u_ic = tst_u_ic./repmat( sqrt(sum(tst_u_ic.^2)), [size(tst_u_ic,1),1]);
    tst_S    = tst_u_ic'*data_tst;
     
    % run LR
    model_trn = QD_NR_train (trn_S, labels, Range);
    map_trn   = QD_map_signed_set (model_trn, trn_u_ic(:,1:Range), trn_S(1:Range,:),'NR' );
    %
    CC=corrcoef([REF_IMG ,map_trn]);
    CC=sign(diag( CC(1:Range,Range+1:end) ));
    map_trn=map_trn*diag(CC);
    
    % run LR
    model_tst = QD_NR_train (tst_S, labels, Range);
    map_tst   = QD_map_signed_set (model_tst, tst_u_ic(:,1:Range), tst_S(1:Range,:),'NR' );
    %
    CC=corrcoef([REF_IMG ,map_tst]);
    CC=sign(diag( CC(1:Range,Range+1:end) ));
    map_tst=map_tst*diag(CC);

    % classify 
    [accur_1] = QD_classify_set (model_trn, trn_u_ic(:,1:Range), trn_avg, data_tst, labels,'NR');        
    [accur_2] = QD_classify_set (model_tst, tst_u_ic(:,1:Range), tst_avg, data_trn, labels,'NR');        
    %
    PP = (accur_1 + accur_2)./2;
    
    % classify 
    [accur_1] = QD_classify_set (model_trn, trn_u_ic(:,1:Range), trn_avg, data_HELD, labels_HELD,'NR');        
    [accur_2] = QD_classify_set (model_tst, tst_u_ic(:,1:Range), tst_avg, data_HELD, labels_HELD,'NR');        
    %
    GG = (accur_1 + accur_2)./2;

end

%% ===================================================================== %%
%%                            SUMMARIZE DATA                             %%
%% ===================================================================== %%

if( strcmp(regtype,'L1') )
    % reproducibility estimation -- truncated for GLASSO estimation
    for(r=1:RangeGLASSO)
        [RR(r) rSPM(:,r)] = get_rSPM( map_trn(:,r), map_tst(:,r), 1 );
    end
else
    % reproducibility estimation
    for(r=1:Range)
        [RR(r) rSPM(:,r)] = get_rSPM( map_trn(:,r), map_tst(:,r), 1 );
    end    
end

DD = sqrt( (1-RR).^2 + (1-PP).^2 );

output.RR   = RR;
output.PP   = PP;
output.GG   = GG;
output.DD   = DD;
output.rSPM = rSPM;
