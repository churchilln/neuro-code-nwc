function [ output ] = SVM_prepare_AND_classify( data_trn, data_tst, REF_IMG, labels, data_HELD, labels_HELD, Range, HiLo, RangeVal2, regtype )
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

if( strcmp(regtype,'PC')  || strcmp(regtype,'IC') )

    KU=3;
    
    rSPM = zeros( Nvox, Range*KU );
    RR   = zeros( Range*KU, 1    );
    PP   = zeros( Range*KU, 1 );

else

    rSPM = zeros( Nvox, Range );
    RR   = zeros( Range, 1    );
    PP   = zeros( Range, 1 );

end

if( strcmp(regtype,'L1') )

    Bound = linspace( HiLo(1), HiLo(2), Range );
    SMS = svmsmoset( 'TolKKT', 1e-6,'KKTViolationLevel',0.05 );

    map_trn = zeros(Nvox,Range);
    map_tst = zeros(Nvox,Range);
    accur_1 = zeros( Range,1 );
    accur_2 = zeros( Range,1 );
    testa_1 = zeros( Range,1 );
    testa_2 = zeros( Range,1 );
    
    for(qq=1:Range)
    svmStruct     = svmtrain(data_trn',labels,'kernel_function', 'linear', 'boxconstraint', 10.^Bound(qq),'method','smo','SMO_OPTS',SMS);
    sind          = svmStruct.SupportVectorIndices;
    map_trn(:,qq) = data_trn(:,sind) * svmStruct.Alpha;
    %
    accur_1(qq) = sum( svmclassify( svmStruct, data_tst' ) == labels' )./length(labels);
    testa_1(qq) = sum( svmclassify( svmStruct, data_HELD' ) == labels_HELD' )./length(labels_HELD);
    end

    for(qq=1:Range)
    svmStruct     = svmtrain(data_tst',labels,'kernel_function', 'linear', 'boxconstraint', 10.^Bound(qq),'method','smo','SMO_OPTS',SMS);
    sind          = svmStruct.SupportVectorIndices;
    map_tst(:,qq) = data_tst(:,sind) * svmStruct.Alpha;
    %
    accur_2(qq) = sum( svmclassify( svmStruct, data_trn' ) == labels' )./length(labels);  
    testa_2(qq) = sum( svmclassify( svmStruct, data_HELD' ) == labels_HELD' )./length(labels_HELD);
    end
    
    CC=corrcoef([REF_IMG ,map_trn]);
    CC=sign(diag( CC(1:Range,Range+1:end) ));
    CC(~isfinite(CC)) = 1;
    map_trn=map_trn*diag(CC);    
    %
    CC=corrcoef([REF_IMG ,map_tst]);
    CC=sign(diag( CC(1:Range,Range+1:end) ));
    CC(~isfinite(CC)) = 1;
    map_tst=map_tst*diag(CC);
    
    % prediction
    PP = (accur_1 + accur_2)./2;
    GG = (testa_1 + testa_2)./2;
    
elseif( strcmp(regtype,'L2') )
    
    Bound = linspace( HiLo(1), HiLo(2), Range );
    
    map_trn = zeros(Nvox,Range);
    map_tst = zeros(Nvox,Range);
    accur_1 = zeros( Range,1 );
    accur_2 = zeros( Range,1 );
    testa_1 = zeros( Range,1 );
    testa_2 = zeros( Range,1 );


    for(qq=1:Range)
    svmStruct     = svmtrain(data_trn',labels,'kernel_function', 'linear', 'boxconstraint', 10.^Bound(qq),'method','qp');%,'autoscale',false);

        if( isempty(svmStruct) )
            map_trn(:,qq) = ones(length(brain_coords),1);
            accur_1(qq)   = 0.00;
            testa_1(qq)   = 0.00;
        else
            sind          = svmStruct.SupportVectorIndices;
            map_trn(:,qq) = data_trn(:,sind) * svmStruct.Alpha;
            accur_1(qq) = sum( svmclassify( svmStruct, data_tst' ) == labels' )./length(labels);
            testa_1(qq) = sum( svmclassify( svmStruct, data_HELD' ) == labels_HELD' )./length(labels_HELD);
        end
    end
disp('plonk');
    for(qq=1:Range)
    svmStruct     = svmtrain(data_tst',labels,'kernel_function', 'linear', 'boxconstraint', 10.^Bound(qq),'method','qp');%,'autoscale',false);

        if( isempty(svmStruct) )
            map_tst(:,qq) = ones(length(brain_coords),1);
            accur_2(qq)=0.00;
            testa_2(qq)=0.00;
        else
            sind          = svmStruct.SupportVectorIndices;
            map_tst(:,qq) = data_tst(:,sind) * svmStruct.Alpha;
            accur_2(qq) = sum( svmclassify( svmStruct, data_trn' ) == labels' )./length(labels);
            testa_2(qq) = sum( svmclassify( svmStruct, data_HELD' ) == labels_HELD' )./length(labels_HELD);
        end
    end
    
    CC=corrcoef([REF_IMG ,map_trn]);
    CC=sign(diag( CC(1:Range,Range+1:end) ));
    CC(~isfinite(CC)) = 1;
    map_trn=map_trn*diag(CC);    
    %
    CC=corrcoef([REF_IMG ,map_tst]);
    CC=sign(diag( CC(1:Range,Range+1:end) ));
    CC(~isfinite(CC)) = 1;
    map_tst=map_tst*diag(CC);
    
    % prediction
    PP = (accur_1 + accur_2)./2;
    GG = (testa_1 + testa_2)./2;
    
elseif( strcmp(regtype,'PC') )
    
    % SVD on full data set (used for reference)
    [v, s] = svd (dataCov_trn); s=sqrt(s); trn_u=data_trn*v*inv(s);
     trn_Z = s*v';
    % SVD on full data set (used for reference)
    [v, s] = svd (dataCov_tst); s=sqrt(s); tst_u=data_tst*v*inv(s);
     tst_Z = s*v';

    % project onto each others' bases:
    trnONtst_Z = tst_u'*data_trn;
    tstONtrn_Z = trn_u'*data_tst;
    % project onto each others' bases:
    heldONtst_Z =  tst_u'*data_HELD;
    heldONtrn_Z = trn_u'*data_HELD;

    Bound = linspace( HiLo(1), HiLo(2), KU );
        
    map_trn = zeros(Nvox,Range*KU);
    map_tst = zeros(Nvox,Range*KU);
    accur_1 = zeros( Range*KU,1 );
    accur_2 = zeros( Range*KU,1 );
    testa_1 = zeros( Range*KU,1 );
    testa_2 = zeros( Range*KU,1 );
    
    kk=0;
    for(ii=1:Range)
    for(jj=1:KU)
        kk=kk+1;
        svmStruct     = svmtrain(trn_Z(1:ii,:)',labels,'kernel_function', 'linear', 'boxconstraint', 10.^Bound(jj),'method','qp');%,'autoscale',false);

        if( isempty(svmStruct) )
            map_trn(:,kk) = zeros(Nvox,1);
            accur_1(kk)=0.00;
            testa_1(kk)=0.00;
        else
            sind          = svmStruct.SupportVectorIndices;
            map_trn(:,kk) = trn_u(:,1:ii) * trn_Z(1:ii,sind) * svmStruct.Alpha;
            %
            accur_1(kk) = sum( svmclassify( svmStruct, tstONtrn_Z(1:ii,:)' ) == labels' )./length(labels);
            testa_1(kk) = sum( svmclassify( svmStruct, heldONtrn_Z(1:ii,:)' ) == labels_HELD' )./length(labels_HELD);
        end
    end
    end

    kk=0;
    for(ii=1:Range)
    for(jj=1:KU)
        kk=kk+1;
        svmStruct     = svmtrain(tst_Z(1:ii,:)',labels,'kernel_function', 'linear', 'boxconstraint', 10.^Bound(jj),'method','qp');%,'autoscale',false);

        if( isempty(svmStruct) )
            map_tst(:,kk) = ones(length(brain_coords),1);
            accur_2(kk)=0.00;
            testa_2(kk)=0.00;
        else
            sind          = svmStruct.SupportVectorIndices;
            map_tst(:,kk) = tst_u(:,1:ii) * trn_Z(1:ii,sind) * svmStruct.Alpha;
            %
            accur_2(kk) = sum( svmclassify( svmStruct, trnONtst_Z(1:ii,:)' ) == labels' )./length(labels);
            testa_2(kk) = sum( svmclassify( svmStruct, heldONtst_Z(1:ii,:)' ) == labels_HELD' )./length(labels_HELD);
        end
    end
    end
    
    CC=corrcoef([REF_IMG ,map_trn]);
    CC=sign(diag( CC(1:kk,kk+1:end) ));    
    CC(~isfinite(CC)) = 1;
    map_trn=map_trn*diag(CC);    
    %
    CC=corrcoef([REF_IMG ,map_tst]);
    CC=sign(diag( CC(1:kk,kk+1:end) ));
    CC(~isfinite(CC)) = 1;
    map_tst=map_tst*diag(CC);
    
    % prediction
    PP = (accur_1 + accur_2)./2;
    GG = (testa_1 + testa_2)./2;
    
elseif(  strcmp(regtype,'IC') )
        
    % SVD on full data set (used for reference)
    [v, s] = svd (dataCov_trn); cusp = sum( diag(s) > 0 );
    s=sqrt(s(1:cusp,1:cusp)); trn_u=data_trn*v(:,1:cusp)*inv(s);
     trn_Z = s*v(:,1:cusp)';
    
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
    [v, s] = svd (dataCov_tst); cusp = sum( diag(s) > 0 );
    s=sqrt(s(1:cusp,1:cusp)); tst_u=data_tst*v(:,1:cusp)*inv(s);
     tst_Z = s*v(:,1:cusp)';
    %
    [tst_S A W] = fastica( tst_Z , 'verbose', 'off','g','tanh', 'numOfIC', RangeVal2 ); 
    lack = RangeVal2 - size(W,1);
    if( lack > 0 ) W = [W; rand( lack,  RangeVal2)]; end
    %
    tst_u_ic = tst_u*W';
    tst_u_ic = tst_u_ic./repmat( sqrt(sum(tst_u_ic.^2)), [size(tst_u_ic,1),1]);
    tst_S    = tst_u_ic'*data_tst;
     
    
    % project onto each others' bases:
    trnONtst_S = tst_u_ic'*data_trn;
    tstONtrn_S = trn_u_ic'*data_tst;
    % project onto each others' bases:
    heldONtst_S = tst_u_ic'*data_HELD;
    heldONtrn_S = trn_u_ic'*data_HELD;
 
    Bound = linspace( HiLo(1), HiLo(2), KU );
 
    map_trn = zeros(Nvox,Range*KU);
    map_tst = zeros(Nvox,Range*KU);
    accur_1 = zeros( Range*KU,1 );
    accur_2 = zeros( Range*KU,1 );
    testa_1 = zeros( Range*KU,1 );
    testa_2 = zeros( Range*KU,1 );

    kk=0;
    for(ii=1:Range)
    for(jj=1:KU)
        kk=kk+1;
        svmStruct     = svmtrain(trn_S(1:ii,:)',labels,'kernel_function', 'linear', 'boxconstraint', 10.^Bound(jj),'method','qp');%,'autoscale',false);

        if( isempty(svmStruct) )
            map_trn(:,kk) = zeros(Nvox,1);
            accur_1(kk)=0.00;
            testa_1(kk)=0.00;
        else
            sind          = svmStruct.SupportVectorIndices;
            map_trn(:,kk) = trn_u_ic(:,1:ii) * trn_S(1:ii,sind) * svmStruct.Alpha;
            %
            accur_1(kk) = sum( svmclassify( svmStruct, tstONtrn_S(1:ii,:)' ) == labels' )./length(labels);
            testa_1(kk) = sum( svmclassify( svmStruct, heldONtrn_S(1:ii,:)' ) == labels_HELD' )./length(labels_HELD);
        end
    end
    end

    kk=0;
    for(ii=1:Range)
    for(jj=1:KU)
        kk=kk+1;
        svmStruct     = svmtrain(tst_S(1:ii,:)',labels,'kernel_function', 'linear', 'boxconstraint', 10.^Bound(jj),'method','qp');%,'autoscale',false);

        if( isempty(svmStruct) )
            map_tst(:,kk) = ones(length(brain_coords),1);
            accur_2(kk)=0.00;
            testa_2(kk)=0.00;
        else
            sind          = svmStruct.SupportVectorIndices;
            map_tst(:,kk) = tst_u_ic(:,1:ii) * trn_S(1:ii,sind) * svmStruct.Alpha;
            %
            accur_2(kk) = sum( svmclassify( svmStruct, trnONtst_S(1:ii,:)' ) == labels' )./length(labels);
            testa_2(kk) = sum( svmclassify( svmStruct, heldONtst_S(1:ii,:)' ) == labels_HELD' )./length(labels_HELD);
        end
    end
    end
    
    CC=corrcoef([REF_IMG ,map_trn]);
    CC=sign(diag( CC(1:kk,kk+1:end) ));
    CC(~isfinite(CC)) = 1;
    map_trn=map_trn*diag(CC);    
    %
    CC=corrcoef([REF_IMG ,map_tst]);
    CC=sign(diag( CC(1:kk,kk+1:end) ));
    CC(~isfinite(CC)) = 1;
    map_tst=map_tst*diag(CC);
    
    % prediction
    PP = (accur_1 + accur_2)./2;
    GG = (testa_1 + testa_2)./2;    
end

%% ===================================================================== %%
%%                            SUMMARIZE DATA                             %%
%% ===================================================================== %%


if( strcmp(regtype,'PC') || strcmp(regtype,'IC') )

    % reproducibility estimation
    for(r=1:round(Range*KU))
        [RR(r) rSPM(:,r)] = get_rSPM( map_trn(:,r), map_tst(:,r), 1 );
        
        if(~isfinite(RR(r)))
            RR(r)=-1.0;
            rSPM(:,r) = zeros(Nvox,1);
        end
    end

else

% reproducibility estimation
    for(r=1:round(Range))
        [RR(r) rSPM(:,r)] = get_rSPM( map_trn(:,r), map_tst(:,r), 1 );
        
        if(~isfinite(RR(r)))
            RR(r)=-1.0;
            rSPM(:,r) = zeros(Nvox,1);
        end
    end

end

DD = sqrt( (1-RR).^2 + (1-PP).^2 );

output.RR   = RR;
output.PP   = PP;
output.GG   = GG;
output.DD   = DD;
output.rSPM = rSPM;
