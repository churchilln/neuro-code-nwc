function [ EIGS ] = LD_prepare_ref_ext( data_ref, labels, Range, Curt, regtype,mask )
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

if( strcmp(regtype,'PCL1') )

    % SVD on full data set (used for reference)
    [ref_u, s, v] = svd (dataCov_ref); s=sqrt(s); ref_u=data_ref*v*inv(s);
     ref_Z = s*v';
     %
     ref_Z = ref_Z(1:Curt,:); ref_u=ref_u(:,1:Curt);
    
    % run LD
    model_ref = LD_L1_train(ref_Z, labels, Range);
    map_ref   = LD_map (model_ref, ref_u);
    %
    CVscores = data_ref' * map_ref; % cv scores {timepts x dims}
    sggn = sign( mean(CVscores( labels==1,: )) - mean(CVscores( labels==0,: )) ); % 
    % now flip-em to match
    map_ref = map_ref * diag(sggn);
    
elseif( strcmp(regtype,'PCL2') )

    % SVD on full data set (used for reference)
    [ref_u, s, v] = svd (dataCov_ref); s=sqrt(s); ref_u=data_ref*v*inv(s);
     ref_Z = s*v';
     %
     ref_Z = ref_Z(1:Curt,:); ref_u=ref_u(:,1:Curt);

    % run LD
    model_ref = LD_L2_train(ref_Z, labels, Range);
    map_ref   = LD_map (model_ref, ref_u);
    %
    CVscores = data_ref' * map_ref; % cv scores {timepts x dims}
    sggn = sign( mean(CVscores( labels==1,: )) - mean(CVscores( labels==0,: )) ); % 
    % now flip-em to match
    map_ref = map_ref * diag(sggn);
    
elseif( strcmp(regtype,'EN') )
    
    % run LD
    model_ref = LD_EN_train( dataCov_ref, labels,Curt, Range );
    map_ref   = LD_map (model_ref, data_ref);
    % flip eigens. to match the CV scores:
    CVscores = data_ref' * map_ref; 
    sggn = sign( mean(CVscores( labels==1,: )) - mean(CVscores( labels==0,: )) ); % 
    % now flip-em to match
    map_ref = map_ref * diag(sggn);

end

EIGS = map_ref;
