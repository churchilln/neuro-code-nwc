function [P, R] = gmm_covariance_generalization( X1, X2, out1, out2 )
%
% =========================================================================
% GMM_COVARIANCE_GENERALIZATION: predictive likelihood on full (voxel x voxel)
% covariance matrix, for clustering algorithms. Uses singular Wishart pdf.
% =========================================================================
%
% Syntax:
%      [ L1 ] = gmm_covariance_generalization( X1, X2, out1, out2, dim )
% 
% Input:
%     (X1,X2) = 3D matrice of input data (V-voxels x T-timepoints x S-subjects)
%               where sth matrix corresponds to the same subject in each
% (out1,out2) = structured outputs from FSIM analysis
%
disp('generalization - singular wishart.dof adjust');

% data dimensions and consistency checking
[dim.V dim.T dim.S] = size(X1);
dim.K = length(out1.pi_k);
if( dim.V~=size(X2,1) || dim.T~=size(X2,2) || dim.S~=size(X2,3) )
    error('input matrix dimensions X1,X2 do not match');
end
if( dim.V~=size(out1.z_kv,2) || dim.T~=size(out1.mu_tks,1) || dim.S~=size(out1.mu_tks,3) || ...
    dim.V~=size(out2.z_kv,2) || dim.T~=size(out2.mu_tks,1) || dim.S~=size(out2.mu_tks,3)   )
    error('gmm outputs do not match dimensions of X1,X2');
end
if( length(out1.pi_k) ~= length(out2.pi_k) )
    error('gmm outputs have different cluster numbers');
end
    
for(split=1:2) %% go through split-halves

    % ----------------- this section defines training parameters

    if( split==1 )

        % estimated by all models
        Z        = out1.z_kv';    % binary LV matrix
        sig2_kvs = out1.sig2_kvs; % noise variance

        %% catch to ensure mean is correctly rescaled
        if( isfield(out1,'H') ) 
              %
              if( isfield(out1,'c_ks') )
                    for(s=1:dim.S) M(:,:,s) = out1.P_s(:,:,s) * out1.H * diag( out1.c_ks(:,s) ); end
              else  for(s=1:dim.S) M(:,:,s) = out1.P_s(:,:,s) * out1.H; end
              end
        else  M  = out1.mu_tks;
        end
        
        % spatial scaling (only in some models)
        if( isfield(out1,'w_kv') ) %% voxel-specific scaling
             %
             vox_scal = out1.w_kv';
        else vox_scal = ones( dim.V, dim.K );
        end

        % test matrix
        X_test  = X2;
        
    elseif( split==2 )

        % estimated by all models
        Z        = out2.z_kv';    % binary LV matrix
        sig2_kvs = out2.sig2_kvs; % noise variance
        
        %% catch to ensure mean is correctly rescaled
        if( isfield(out2,'H') ) 
              %
              if( isfield(out2,'c_ks') )
                    for(s=1:dim.S) M(:,:,s) = out2.P_s(:,:,s) * out2.H * diag( out2.c_ks(:,s) ); end
              else  for(s=1:dim.S) M(:,:,s) = out2.P_s(:,:,s) * out2.H; end
              end
        else  M  = out2.mu_tks;
        end

        % spatial scaling (only in some models)
        if( isfield(out2,'w_kv') ) %% voxel-specific scaling
             %
             vox_scal = out2.w_kv';
        else vox_scal = ones( dim.V, dim.K );
        end
        
        % test matrix
        X_test  = X1;
    end
    
%% Predictive likelihood on covariance:

    for( s=1:dim.S ) %% iterate through subjects

        [split s],
        
        % bookkeeping... dof and log-mv-gamma
        T_dof = rank( X_test(:,:,s) );
        lmgam = logMvGamma( 0.5*T_dof, T_dof);
        
        % (v x t) matrix of "noise-free" voxel timeseries, under clustering model
        % each voxel is represented by the (scaled) cluster mean it is assigned to it
        X_sig  = (Z .* vox_scal)*M(:,:,s)';
        % eigendecomposition of matrix --> reduce because there can only be K non-rank-deficient components
        [U L] = svd( X_sig, 'econ' ); 
        U=U(:,1:dim.K); L=L(1:dim.K,1:dim.K);
        
        % used to get features of SVD for "large" (v x v) covariance, S_sig = U*diag(D_sig)*U'
        % vector of signal variance for each component (k x 1), unnormalized
        D_sig = diag( L.^2 );
        % vector of noise variance for each voxel (v x 1):
        D_noi = sum(Z.*sig2_kvs(:,:,s)',2); %% cluster/voxel/subject specific variance

        % terms re-used in likelihood:
        UL_sig  = bsxfun(@times, U, sqrt(D_sig)'); % weight components by kth eig.value
        XtDinvU = X_test(:,:,s)'* bsxfun(@times, U, 1./D_noi   ); % x'*(Dn-1 * U) --> voxel-weighting of components

        % log-likelihood:   0.5*( T**(T* - V)*log(pi) - T**V*log(2) ) - lmgam
        %                  +0.5*(T*-V-1)*log(det(X*X')_+)
        %                  -0.5*( T'*log(det(Sig)) + tr(X'*inv(Sig)*X) ) 
        %
        % terms independent of X and Sigma
        Fixed_terms = 0.5*( T_dof*(T_dof - dim.V)*log(pi) - T_dof*dim.V*log(2) ) - lmgam;
        % term fixed in X
        [u l2]      = svd( X_test(:,:,s),'econ' ); l2=diag(l2.^2); l2=l2(1:T_dof); % take non-zero sing.val^2
        XX_logdet   =  0.5 * (T_dof - dim.V - 1) * sum(log(l2)); %sum of log-(sing.val)^2
        % Sig-term1: log-det with T_dof multiplier:
        Sig_logdet  = -0.5 * T_dof * (  dim.V*(log(dim.T-T_dof)) + sum(log(D_noi)) + log(det( eye(dim.K) + UL_sig'*bsxfun(@rdivide,UL_sig, dim.T*D_noi) ))  );
        % Sig-term2: cross-terms between X and Sigma:
        Sig_tracex  = -0.5 * (T_dof/dim.T) * (  sum(sum( bsxfun(@times, X_test(:,:,s).^2, 1./D_noi ),1 ),2)  -  (trace( XtDinvU* inv( diag(1./D_sig) + (U'*bsxfun(@times, U, 1./D_noi))./dim.T ) *XtDinvU' )./(dim.T))  );
        % log-likelihood of terms
        llik1(s,1)  = Fixed_terms + XX_logdet + Sig_logdet + Sig_tracex;
    end

    % total log-likelihood over all timepoints, subjects
    L1(:,split) = llik1(:);
end

% prediction: average log-likelihood across split-halves
P=mean(L1,2);
% reproducibility: estimated normalized mutual information
R = nmi( out1.z_kv'*(1:dim.K)', out2.z_kv'*(1:dim.K)' );

%%
function v = nmi(x, y)
% Nomalized mutual information
% adapted from Mo Chen (mochen@ie.cuhk.edu.hk). March 2009.

x = x(:);
y = y(:);
x_unique = unique(x);
y_unique = unique(y);
n = length(x);

% check the integrity of y
if length(x_unique) ~= length(y_unique)
    error('The clustering y is not consistent with x.');
end;

% distribution of y and x
Ml = double( bsxfun(@eq,x,x_unique') );
Mr = double( bsxfun(@eq,y,y_unique') );
% probability per unique element
Pl = sum(Ml,1)/n;
Pr = sum(Mr,1)/n;
% entropy of Pr and Pl
Hl = -sum( Pl .* log2( Pl + eps ) );
Hr = -sum( Pr .* log2( Pr + eps ) );
% joint entropy of Pr and Pl
M   = Ml'*Mr/n;
Hlr = -sum( M(:) .* log2( M(:) + eps ) );
% mutual information
MI = Hl + Hr - Hlr;
% normalized mutual information
v = sqrt((MI/Hl)*(MI/Hr)) ;


%%
function y = logMvGamma(x,d)
% Compute logarithm multivariate Gamma function.
% Gamma_p(x) = pi^(p(p-1)/4) prod_(j=1)^p Gamma(x+(1-j)/2)
% log Gamma_p(x) = p(p-1)/4 log pi + sum_(j=1)^p log Gamma(x+(1-j)/2)
% Written by Michael Chen (sth4nth@gmail.com).
s = size(x);
x = reshape(x,1,prod(s));
x = bsxfun(@plus,repmat(x,d,1),(1-(1:d)')/2);
y = d*(d-1)/4*log(pi)+sum(gammaln(x),1);
y = reshape(y,s);