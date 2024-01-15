function [ EIGS ] = QD_prepare_ref( data_ref, labels, Range, RangeVal2, regtype,mask )
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

    FirstLim = Range+1;
    
    % run LR
    model_ref = QD_L1_train (dataCov_ref, labels, FirstLim, Range);
    map_ref   = QD_map_signed_set (model_ref, data_ref, dataCov_ref, 'L1' );
    % flip eigens. to match the CV scores:
    CVscores = data_ref' * map_ref; 
    sggn = sign( mean(CVscores( labels==1,: )) - mean(CVscores( labels==0,: )) ); % 
    % now flip-em to match
    map_ref = map_ref * diag(sggn);
        
elseif( strcmp(regtype,'L2') )

    % run LR
    model_ref = QD_L2_train (dataCov_ref, labels, Range);
    map_ref   = QD_map_signed_set (model_ref, data_ref, dataCov_ref, 'L2' );
    % flip eigens. to match the CV scores:
    CVscores = data_ref' * map_ref; 
    sggn = sign( mean(CVscores( labels==1,: )) - mean(CVscores( labels==0,: )) ); % 
    % now flip-em to match
    map_ref = map_ref * diag(sggn);

elseif( strcmp(regtype,'PC') )
    
    % SVD on full data set (used for reference)
    [ref_u, s, v] = svd (dataCov_ref); s=sqrt(s); ref_u=data_ref*v*inv(s);
     ref_Z = s*v';
    % run LR
    model_ref = QD_NR_train (ref_Z, labels, Range);
    map_ref   = QD_map_signed_set (model_ref, ref_u(:,1:Range), ref_Z(1:Range,:),'NR' );
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

    % >>> dims x lambda matrix of discriminants
    model_ref = QD_NR_train (ref_S, labels, Range);
    map_ref   = QD_map_signed_set (model_ref, ref_u_ic(:,1:Range), ref_S(1:Range,:),'NR' );

        % flip to match the CV scores:
    CVscores = data_ref' * map_ref; % cv scores {timepts x dims}
    sggn = sign( mean(CVscores( labels==1,: )) - mean(CVscores( labels==0,: )) ); % 
    % now flip-em to match
    map_ref = map_ref * diag(sggn);

end

EIGS = map_ref;
