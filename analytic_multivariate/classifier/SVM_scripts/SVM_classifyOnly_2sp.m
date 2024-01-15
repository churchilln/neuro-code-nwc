function [ GG ] = SVM_classifyOnly_2sp( data_trn, data_tst, labels, data_HELD, labels_HELD, Range, HiLo, RangeVal2, regtype )
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
    GG   = zeros( Range*KU, 1 );
else
    GG   = zeros( Range, 1 );
end

if( strcmp(regtype,'L1') )

    Bound = linspace( HiLo(1), HiLo(2), Range );
    SMS = svmsmoset( 'TolKKT', 1e-6,'KKTViolationLevel',0.05 );

    for(qq=1:Range)
    svmStruct   = svmtrain(data_trn',labels,'kernel_function', 'linear', 'boxconstraint', 10.^Bound(qq),'method','smo','SMO_OPTS',SMS);
    accur_1(qq) = sum( svmclassify( svmStruct, data_HELD' ) == labels_HELD' )./length(labels_HELD);
    end

    for(qq=1:Range)
    svmStruct     = svmtrain(data_tst',labels,'kernel_function', 'linear', 'boxconstraint', 10.^Bound(qq),'method','smo','SMO_OPTS',SMS);
    accur_2(qq) = sum( svmclassify( svmStruct, data_HELD' ) == labels_HELD' )./length(labels_HELD);
    end
        
    % prediction
    GG = (accur_1 + accur_2)./2;
    
elseif( strcmp(regtype,'L2') )
    

    Bound = linspace( HiLo(1), HiLo(2), Range );

    for(qq=1:Range)
    svmStruct     = svmtrain(data_trn',labels,'kernel_function', 'linear', 'boxconstraint', 10.^Bound(qq),'method','qp');%,'autoscale',false);

        if( isempty(svmStruct) )
            accur_1(qq)   = 0.49;
        else
            accur_1(qq) = sum( svmclassify( svmStruct, data_HELD' ) == labels_HELD' )./length(labels_HELD);
        end
    end
disp('plonk');
    for(qq=1:Range)
    svmStruct     = svmtrain(data_tst',labels,'kernel_function', 'linear', 'boxconstraint', 10.^Bound(qq),'method','qp');%,'autoscale',false);

        if( isempty(svmStruct) )
            accur_2(qq)   = 0.49;
        else
            accur_2(qq) = sum( svmclassify( svmStruct, data_HELD' ) == labels_HELD' )./length(labels_HELD);
        end
    end
    
    % prediction
    GG = (accur_1 + accur_2)./2;
    
elseif( strcmp(regtype,'PC') )
    
    % SVD on full data set (used for reference)
    [trn_u, s, v] = svd (dataCov_trn); s=sqrt(s); trn_u=data_trn*v*inv(s);
     trn_Z = s*v';
    % SVD on full data set (used for reference)
    [tst_u, s, v] = svd (dataCov_tst); s=sqrt(s); tst_u=data_tst*v*inv(s);
     tst_Z = s*v';

    % project onto each others' bases:
    heldONtst_Z =  tst_u'*data_HELD;
    heldONtrn_Z = trn_u'*data_HELD;
    
    Bound = linspace( HiLo(1), HiLo(2), Range );
        
    kk=0;
    for(ii=1:Range)
    for(jj=1:KU)
        kk=kk+1;
        svmStruct     = svmtrain(trn_Z(1:ii,:)',labels,'kernel_function', 'linear', 'boxconstraint', 10.^Bound(jj),'method','qp');%,'autoscale',false);

        if( isempty(svmStruct) )
            accur_1(kk)=0.49;
        else
            accur_1(kk) = sum( svmclassify( svmStruct, heldONtrn_Z(1:ii,:)' ) == labels_HELD' )./length(labels_HELD);
        end
    end
    end

    kk=0;
    for(ii=1:Range)
    for(jj=1:KU)
        kk=kk+1;
        svmStruct     = svmtrain(tst_Z(1:ii,:)',labels,'kernel_function', 'linear', 'boxconstraint', 10.^Bound(jj),'method','qp');%,'autoscale',false);

        if( isempty(svmStruct) )
            accur_2(kk)=0.49;
        else
            accur_2(kk) = sum( svmclassify( svmStruct, heldONtst_Z(1:ii,:)' ) == labels_HELD' )./length(labels_HELD);
        end
    end
    end
        
    % prediction
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
    [tst_S A W] = fastica( tst_Z , 'verbose', 'off','g','tanh', 'numOfIC', RangeVal2 ); 
    lack = RangeVal2 - size(W,1);
    if( lack > 0 ) W = [W; rand( lack,  RangeVal2)]; end
    %
    tst_u_ic = tst_u*W';
    tst_u_ic = tst_u_ic./repmat( sqrt(sum(tst_u_ic.^2)), [size(tst_u_ic,1),1]);
    tst_S    = tst_u_ic'*data_tst;
     
    
    % project onto each others' bases:
    heldONtst_S = tst_u_ic'*data_HELD;
    heldONtrn_S = trn_u_ic'*data_HELD;
     
    Bound = linspace( HiLo(1), HiLo(2), Range );
        
    kk=0;
    for(ii=1:Range)
    for(jj=1:KU)
        kk=kk+1;
        svmStruct     = svmtrain(trn_S(1:ii,:)',labels,'kernel_function', 'linear', 'boxconstraint', 10.^Bound(jj),'method','qp');%,'autoscale',false);

        if( isempty(svmStruct) )
            accur_1(kk)=0.49;
        else
            accur_1(kk) = sum( svmclassify( svmStruct, heldONtrn_S(1:ii,:)' ) == labels_HELD' )./length(labels_HELD);
        end
    end
    end

    kk=0;
    for(ii=1:Range)
    for(jj=1:KU)
        kk=kk+1;
        svmStruct     = svmtrain(tst_S(1:ii,:)',labels,'kernel_function', 'linear', 'boxconstraint', 10.^Bound(jj),'method','qp');%,'autoscale',false);

        if( isempty(svmStruct) )
            accur_2(kk)=0.49;
        else
            accur_2(kk) = sum( svmclassify( svmStruct, heldONtst_S(1:ii,:)' ) == labels_HELD' )./length(labels_HELD);
        end
    end
    end
        
    % prediction
    GG = (accur_1 + accur_2)./2;
    
end

%% ===================================================================== %%
%%                            SUMMARIZE DATA                             %%
%% ===================================================================== %%
