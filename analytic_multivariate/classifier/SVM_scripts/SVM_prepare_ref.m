function [ EIGS ] = SVM_prepare_ref( data_ref, labels, Range, HiLo, RangeVal2, regtype,mask )
%
% .For now, assume DATA_cl0/DATA_cl1 are equal sized (3d matrixes)
%

% matrix dimensions
[Nvox Nsamp] = size( data_ref );
% full data matrix, mean centered -- then covariance!
data_ref    = data_ref - repmat(mean(data_ref,2),[1 Nsamp]);
dataCov_ref = data_ref'*data_ref;

%% ===================================================================== %%
%%                              TRAIN DATA                               %%
%% ===================================================================== %%

if( strcmp(regtype,'L1') )

    Bound = linspace( HiLo(1), HiLo(2), Range );
    SMS = svmsmoset( 'TolKKT', 1e-6,'KKTViolationLevel',0.05 );
    
    map_ref = zeros(Nvox,Range);
    
    for(qq=1:Range)
    svmStruct     = svmtrain(data_ref',labels,'kernel_function', 'linear', 'boxconstraint', 10.^Bound(qq),'method','smo','SMO_OPTS',SMS);
    sind          = svmStruct.SupportVectorIndices;
    map_ref(:,qq) = data_ref(:,sind) * svmStruct.Alpha;    
    end
    
    % flip eigens. to match the CV scores:
    CVscores = data_ref' * map_ref; 
    sggn = sign( mean(CVscores( labels==1,: )) - mean(CVscores( labels==0,: )) ); % 
    % now flip-em to match
    map_ref = map_ref * diag(sggn);
    
elseif( strcmp(regtype,'L2') )

    Bound = linspace( HiLo(1), HiLo(2), Range );
    
    map_ref = zeros(Nvox,Range);
    
    for(qq=1:Range)
    svmStruct     = svmtrain(data_ref',labels,'kernel_function', 'linear', 'boxconstraint', 10.^Bound(qq),'method','qp');%,'autoscale',false);
    
        if( isempty(svmStruct) )
            map_ref(:,qq) = zeros(Nvox,1);
        else
            sind          = svmStruct.SupportVectorIndices;
            map_ref(:,qq) = data_ref(:,sind) * svmStruct.Alpha;
        end
    end
    
    % flip eigens. to match the CV scores:
    CVscores = data_ref' * map_ref; 
    sggn = sign( mean(CVscores( labels==1,: )) - mean(CVscores( labels==0,: )) ); % 
    % now flip-em to match
    map_ref = map_ref * diag(sggn);

elseif( strcmp(regtype,'PC') )
    
    KU=3;
    
    % SVD on full data set (used for reference)
    [ref_u, s, v] = svd (dataCov_ref); s=sqrt(s); ref_u=data_ref*v*inv(s);
     ref_Z = s*v';
       
    Bound = linspace( HiLo(1), HiLo(2), KU );
    
    map_ref = zeros(Nvox,Range*KU);
    
    kk=0;
    for(ii=1:Range)
    for(jj=1:KU)
    
        kk=kk+1;
        svmStruct     = svmtrain(ref_Z(1:ii,:)',labels,'kernel_function', 'linear', 'boxconstraint', 10.^Bound(jj),'method','qp');%,'autoscale',false);
        if( isempty(svmStruct) )
            map_ref(:,kk) = zeros(Nvox,1);
        else
            sind          = svmStruct.SupportVectorIndices;
            map_ref(:,kk) = ref_u(:,1:ii) * ref_Z(1:ii,sind) * svmStruct.Alpha;
        end
    end
    end

    % flip eigens. to match the CV scores:
    CVscores = data_ref' * map_ref; % cv scores {timepts x dims}
    sggn = sign( mean(CVscores( labels==1,: )) - mean(CVscores( labels==0,: )) ); % 
    % now flip-em to match
    map_ref = map_ref * diag(sggn);

elseif(  strcmp(regtype,'IC') )
        
    % SVD on full data set (used for reference)
    [ref_u, s, v] = svd (dataCov_ref); s=sqrt(s); ref_u=data_ref*v*inv(s);
     ref_Z = s*v';
    % trim dimensionality
     ref_Z = ref_Z(1:RangeVal2,:);
     ref_u = ref_u(:,1:RangeVal2);
    
    [ref_S A W] = fastica( ref_Z , 'verbose', 'off','g','tanh', 'numOfIC', RangeVal2 );
    lack = RangeVal2 - size(W,1);
    if( lack > 0 ) W = [W; rand( lack,  RangeVal2)]; end
    %
    ref_u_ic = ref_u*W';
    ref_u_ic = ref_u_ic./repmat( sqrt(sum(ref_u_ic.^2)), [size(ref_u_ic,1),1]);
    ref_S    = ref_u_ic'*data_ref;
    
    KU=3;
       
    Bound = linspace( HiLo(1), HiLo(2), KU );
    
    map_ref = zeros(Nvox,Range*KU);
    
    kk=0;
    for(ii=1:Range)
    for(jj=1:KU)
    
        kk=kk+1;
        svmStruct     = svmtrain(ref_S(1:ii,:)',labels,'kernel_function', 'linear', 'boxconstraint', 10.^Bound(jj),'method','qp');%,'autoscale',false);
        if( isempty(svmStruct) )
            map_ref(:,kk) = zeros(Nvox,1);
        else
            sind          = svmStruct.SupportVectorIndices;
            map_ref(:,kk) = ref_u_ic(:,1:ii) * ref_S(1:ii,sind) * svmStruct.Alpha;
        end
    end
    end

    % flip eigens. to match the CV scores:
    CVscores = data_ref' * map_ref; % cv scores {timepts x dims}
    sggn = sign( mean(CVscores( labels==1,: )) - mean(CVscores( labels==0,: )) ); % 
    % now flip-em to match
    map_ref = map_ref * diag(sggn);


end

EIGS = map_ref;
