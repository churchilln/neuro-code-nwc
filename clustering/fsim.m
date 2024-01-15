function [ outset ] = fsim( X, K, varargin )
%
% =========================================================================
% FSIM: clustering model for group fMRI data, which learns the ML (or MAP) 
% voxel clusters, under the "PARAFAC2" constraint. This requires that 
% subjects have a common network covariance, with possible 
% subject-specific scaling parameters. The model is developed and
% implemented in an Expectation-Maximization (EM) framework.
% The model also allows you to adjust parameter flexibility as desired
% (see below for further details).
% =========================================================================
%
%  SYNTAX:
%          output = fsim( X, K, (MAX_ITER, type, init, MSTEPS, Prior, FixedParc) )
%
%   INPUT:
%          X        = 3D matrix of input data (V-voxels x T-timepoints x S-subjects)
%          K        = number of clusters (scalar value)
%          MAX_ITER = maximum number of iterations before termination
%                     * optional field, default MAX_ITER=100
%                     * otherwise terminates when change in log-likelihood < 1E-10, and stable parcellations
%          type     = determine level of mixture model flexibility. Options
%                     ordered by increasing model complexity:
%                       type=1: standard GMM
%                       type=2: adaptive voxel scalings w_kv
%                       type=3: common covariance H'*H (and w_kv)
%                       type=4: network scaling c_ks (and w_kv, H'*H)
%                     *optional field, default type=4 
%          init     = determines initialization method. Four options right now:
%                       init= (Kx1) vector of integers:  list of seed voxel indices (corresponding to dim.1 of X) to initialize clusters
%                       init= (KxV) binary matrix: predefined parcellations
%                       init=  1:  initialize by randomly selecting K "seed" voxels
%                       init=  2:  initialize by k-means clustering (best of 10 random starts)
%                     *optional field, default init=2
%          MSTEPS   = number of M-step iterations, per EM cycle
%                     (usually MSTEPS=1 works fine)
%                     *optional field, default MSTEPS=1
%          Prior    = prior placed on noise variance to avoid model collapse due to singular
%                     clusters (i.e. a single datapoint with zero variance)
%                        0 =       no prior: this WILL cause the model to collapse for most 
%                                            high-dimensional fmri data. Not recommended unless you have good reason! 
%                        1 = improper prior: weak prior, minimizes model bias
%                        2 =  inverse gamma: a "true" conjugate prior to Gaussian model, but heavier bias 
%                     *optional field, default Prior=1
%          FixedParc = cluster responsibilities are held fixed (no E-step updating)
%                      *optional field, default FixedParc=0
%
%  OUTPUT: 
%
%         output.gamm_kv  = (K x V) matrix of voxel responsibilities for each cluster;
%         output.pi_k     = (K x 1) vector of cluster priors
%         output.mu_tks   = (T x K x S) matrix of cluster mean vectors
%         output.sig2_kvs = (K x V x S) matrix of cluster/voxel/subject noise variance
%
%         output.w_kv     = (K x V) matrix of cluster-wise voxel scaling,
%                           normalized to unit-length for each cluster (only for model types 2,3,4)
%         output.H        = (K x K) matrix used to define common covariance COV = H'*H (only for model types 3,4)
%         output.P_s      = (T x K x S) matrix of subjects' orthonormal timeseries (only for model types 3,4) 
%         output.c_ks     = (K x V) matrix of cluster-wise subject scaling (only for model type 4)
%
%         output.LLIK     = (MAX_ITER x 1) vector, showing log-likelihood per iteration
%         output.init     = (K x 1) vector of seed voxels used to initialize EM,
%                            if seed was pre-specified, or random-start used
% 
%  --------------------------
%  ver. update: 2015/07/03
%  --------------------------
%

%% (0) interpreting input arguments

disp('Preparing to cluster...');

nvararg = length(varargin);
if( nvararg<1 || isempty(varargin{1}) ) %number of iterations
     MAX_ITER=100;
else MAX_ITER=varargin{1};
end
if( nvararg<2 || isempty(varargin{2}) ) %GMM model type
     type=4;
else type=varargin{2};
end
if( nvararg<3 || isempty(varargin{3}) ) %cluster initialization
     init=2;
else init=varargin{3};
end
if( nvararg<4 || isempty(varargin{4}) ) %number of m-step updates
     MSTEPS=1;
else MSTEPS=varargin{4};
end
if( nvararg<5 || isempty(varargin{5}) ) %variance prior
     dim.PRIOR=1;
else dim.PRIOR=varargin{5};
end
if( nvararg<6 || isempty(varargin{6}) ) %e-step updating turned off?
     FIXPARC=0;
else FIXPARC=varargin{6};
end

if( FIXPARC>0 && numel(init)>1 && sum(size(init)>1)>1 ) % validity check --> if passed, FIXPARC is used to flag special cases
    disp('fixed parcellation allowed, parcel matrix is valid...');
elseif( FIXPARC>0 )
    error('can only do fixed parcellation with a predefined parcel matrix!');
end

if( numel(init)>1 && sum(size(init)>1)>1 ) initdisp = 'parcel';
elseif( numel(init)>1 && sum(size(init)>1)==1 ) initdisp = 'seeds';
else initdisp=num2str(init);
end

disp(['model settings: MAX_ITER=',num2str(MAX_ITER),' type=',num2str(type),' init=[',initdisp,'] MSTEPS=',num2str(MSTEPS),' PRIOR=',num2str(dim.PRIOR),' FIXPARC=',num2str(FIXPARC)]);

%% (1) cluster seed initialization

% dimensions of input data matric (V-voxels x T-timepoints x S-subjects)
[dim.V, dim.T, dim.S] = size( X ); 
% also store K in "dim" structure
dim.K=K; clear K; 

% ----- Initializing cluster means ----- %
%
if( numel(init)>1 && sum(size(init)>1)>1 ) %% using pre-specified parcels

    seed = init; % rename seed array
    % catch if incorrect #seed vectors specified
    if( size(seed,1) ~= dim.K )
        error('pre-specified seed matrix is incorrect length (dim-K)');
    end
    if( size(seed,2) ~= dim.V )
        error('pre-specified seed matrix is incorrect length (dim-V)');
    end
    % initialize with pre-specified parcel matrix
    for(k=1:dim.K)
        MU_init(:,k,:) = permute( mean( X(seed(k,:)>0,:,:), 1),  [2 3 1]);
    end
    
elseif( numel(init)>1 && sum(size(init)>1)==1 ) %% using pre-specified seed

    seed = init; % rename seed array
    % catch if incorrect #seed vectors specified
    if( length(seed) ~= dim.K )
        error('pre-specified seed is incorrect length');
    end
    % initialize with pre-specified seed vector
    MU_init = permute( X(seed,:,:), [2 1 3] );

elseif( init==1 )    %% using random initialization
    %
    % random selection of k voxels
    list = randperm(dim.V);
    seed = list(1:dim.K);
    %% use random seed voxels as starting point
    MU_init = permute( X(seed,:,:), [2 1 3] );
    
elseif( init==2 )    %% using k-means initialization (best-of-10)
    %
    % run kmeans analysis
    [out] = kmeans_group( X , dim.K, 10);
    %% use 10-iter optimal kmeans as starting point
    MU_init = permute( out.mu_kts, [2 1 3] );
    %
    seed = 'k-means';
else
    error('invalid seed option.');
end
clear init;

%% (2) initializing model parameters

% initializing noise variance prior (stored in "lambda" variable)
if(dim.PRIOR==1)
    % Improper prior (negative exponential) --> 1 param.
    lambda = 1E-6;
elseif(dim.PRIOR==2)
    % Inverse Gamma prior --> 1 param.
    lambda = [1E-6 mean(mean(var(X,0,2)))];
end

% fixed parameters
if(FIXPARC>0) gamm_kv = seed; %%% Fixed Parcellation (no E-step)
else          gamm_kv = ones( dim.K,dim.V )./dim.K; % all clusters have equal responsibility per voxel
end
pi_k      = ones( dim.K,1 )./dim.K;     % all clusters are equally probable
sig2_kvs  = ones( dim.K,dim.V,dim.S);   % cluster/subject variance --> unit variance
w_kv      = ones( dim.K,dim.V   );      % cluster/voxel scaling --> unit scaling
% initial cluster means, to "jump-start" model
mu_tks    = MU_init;                    % cluster means --> use initialization
scal      = sqrt(sum(mu_tks.^2,1));     % obtain sqared scaling of means
% for common covariance, initialize sub-components of the mean
if(type>=3)
P_s       = bsxfun(@rdivide, mu_tks, scal);  % unit-normed cluster means (non-orthogonal!)
H         = eye( dim.K );               % produces fixed covariance matrix, independent components
end
% for network scaling, initialize terms
if(type==4) c_ks      = permute(scal,[2 3 1]);           % cluster/subject scaling = scale of current means
else        c_ks      = ones( dim.K,dim.S   );
end    
% these settings ensure "mu(s) = P(s)*H*diag(c_k(s))" relationship is preserved

%% (3) Standard GEM optimization

%initialization
iter=0; terminflag = false;

while( iter<MAX_ITER && ~terminflag ) % continue intil MAX_ITER (no stopping criterion yet)
    
    iter=iter+1; % increment
    
    tic
    % display iteration #
    disp(['iteration ',num2str(iter),' of ',num2str(MAX_ITER)]);

    % =====================================================================
    % (E-step): update voxel responsibilities, and full data log-likelihood  
    % =====================================================================
    
    % get log-likelihood and responsibilities
    if(FIXPARC>0) [~,       LLIK(iter,1)] = ESTEP_gamm_kv_and_LLIK(X, pi_k, mu_tks, sig2_kvs, w_kv, dim ); %%% Fixed Parcellation (no E-step)
    else          [gamm_kv, LLIK(iter,1)] = ESTEP_gamm_kv_and_LLIK(X, pi_k, mu_tks, sig2_kvs, w_kv, dim ); 
    end
    
    if(dim.PRIOR==1)
        % prior-adjusted likelihood, improper (negative exponential)
        LLIK(iter,1) = LLIK(iter,1) - lambda.*( sum(sum(sum(log(sig2_kvs)))) + sum(sum(sum(1./sig2_kvs))) );
    elseif(dim.PRIOR==2)
        % prior-adjusted likelihood, inverse gamma
        LLIK(iter,1) = LLIK(iter,1) - ( (lambda(1)+1)*sum(sum(sum(log(sig2_kvs)))) + sum(sum(sum(lambda(2)./sig2_kvs))) );
    end
    
    % =====================================================================
    % (M-step): updating each clustering parameter in turn
    % =====================================================================

    % cluster priors, independent of other parameters
    pi_k    = MSTEP_pi_k( gamm_kv, dim );
    
    for(miter=1:MSTEPS) %% number of M-step cycles (inter-dependent steps)

        if(type>=2) 
            % voxel scaling updated
            w_kv    = MSTEP_w_kv( X, mu_tks, sig2_kvs, dim );                      
        end
        
        if(type>=3) 
            % common covariance updated
            [P_s, H] = MSTEP_P_s_and_H( X, c_ks, sig2_kvs, w_kv, gamm_kv, H, dim );
            %
            if(type==4) 
                % network scaling update
                c_ks = MSTEP_c_ks( X, sig2_kvs, w_kv, gamm_kv, H, P_s, dim );
            end       
            % update cluster means based on components
            for(s=1:dim.S) mu_tks(:,:,s) = P_s(:,:,s) * H * diag( c_ks(:,s) ); end
        else
            % otherwise update subject/cluster means and variance independently
            mu_tks = MSTEP_mu_tks( X, sig2_kvs, gamm_kv, w_kv, dim );
        end  
        % update variance estimate
        sig2_kvs = MSTEP_sig2_kvs( X, gamm_kv, mu_tks, w_kv, lambda, dim );           
    end
    
    if(iter>1) %% Convergence testing. Only after at least 1 EM cycle.

        % Test 1: fraction of voxels with a new cluster assignment 
        % assumes approximately binary class assignments. Counts fraction of voxels where p=1 entries don't match
        frac_gamm(iter,1) = mean( sum( double(gamm_kv>0.5) .* double(gamm_kv_OLD>0.5), 1 ) == 0,2 );
        % Replace current "old" gamma with the updated version
        gamm_kv_OLD = gamm_kv;

        % Test 2: fractional change in log-likelihood
        frac_llik(iter,1) = abs( (LLIK(iter) - LLIK(iter-1))./LLIK(iter-1) );
    else
        % because iter-1 has nothing to compare against
        frac_gamm(iter,1) = Inf;
        frac_llik(iter,1) = Inf;
        % initialize "old" parcellation map, to test for convergence
        gamm_kv_OLD = gamm_kv;        
    end
    toc
    
    % check stopping condition(s)
    if( iter>1 && frac_gamm(iter)==0 && frac_llik(iter)< 1E-6 )
       disp('early termination.');
       terminflag=true;
    end
end

% trim 

%% (4) storing outputs

% component renormalization for model types 2-4
if    (type>=2)    
    %
    % norm of each "scaling" image (kx1)
    nrm_w  = sqrt(sum(w_kv.^2,2));
    % unit-normalize spatial scaling
    w_kv   = bsxfun(@rdivide, w_kv, nrm_w);
    
    if    (type==2) %% renorm (w_kv)
        %
        % "Scale up" the mean subject timeseries
        mu_tks = bsxfun(@times, mu_tks, nrm_w' );

    elseif(type==3) %% renorm (w_kv)
        %
        % "Scale up" group covariance matrix
        H = bsxfun(@times, H, nrm_w');
        % redefine cluster means, based on voxel rescaling:
        for(s=1:dim.S) mu_tks(:,:,s) = P_s(:,:,s) * H * diag( c_ks(:,s) ); end 

    elseif(type==4) %% renorm (w_kv,c_ks)
        %
        nrm_H = sqrt(sum(H.^2,1));    % norm of each common basis vector (1xk)
        % unit-normalize group covariance matrix
        H     = bsxfun(@rdivide, H,    nrm_H);
        % "Scale up" the subject scaling components
        c_ks  = bsxfun(@times, c_ks, nrm_w.*nrm_H');
        % redefine cluster means, based on voxel rescaling:
        for(s=1:dim.S) mu_tks(:,:,s) = P_s(:,:,s) * H * diag( c_ks(:,s) ); end    
    end
end

% binarized latent variables
z_kv = double( bsxfun(@rdivide, gamm_kv, max(gamm_kv,[],1)) == 1);

% storing results:
%
% parameters common to all models
outset.gamm_kv  =  gamm_kv;
outset.z_kv     =     z_kv;
outset.pi_k     =     pi_k;
outset.mu_tks   =   mu_tks;
outset.sig2_kvs = sig2_kvs;
% parameters specific to different models 
if(type>=2) %%spatial scaling
    outset.w_kv = w_kv;
    if(type>=3) %%common covariance
        outset.H   =   H;
        outset.P_s = P_s;
        if(type==4) %%network scaling
            outset.c_ks = c_ks;
        end
    end
end
% likelihood stats
outset.LLIK    =    LLIK;
outset.init    =    seed;
outset.type    =    type;
% convergence statistics
outset.frac_gamm = frac_gamm; % fractional change in cluster assign.
outset.frac_llik = frac_llik; % fractional change in log-likelihood

%% ====================================================== %%
%%    DEFINE INDIVIDUAL E-STEP, M-STEP FUNCTIONS BELOW    %%
%% ====================================================== %%

%% [voxel responsibilities + total model log-likelihood]
function [x1, x2] = ESTEP_gamm_kv_and_LLIK(X, pi_k, mu_tks, sig2_kvs, w_kv, dim )

Q = zeros( dim.V, dim.K );
for(s=1:dim.S)
    Q = Q + bsxfun(@plus,sum(X(:,:,s).^2,2), -2*(w_kv').*(X(:,:,s)*mu_tks(:,:,s))+bsxfun(@times,(w_kv.^2)',sum(mu_tks(:,:,s).^2)) ) ./ sig2_kvs(:,:,s)';
end
llik_vk = -0.5.*( dim.T.*sum(log(2*pi.*sig2_kvs),3)'+ Q ); 
% total set of (v x k) joint likelihoods (add log-prior)
A0=bsxfun(@plus,log(pi_k)',llik_vk);
% max log-likelihood across clusters, for each voxel
maxA0=max(A0,[],2);
% cluster likelihoods via log-sum-exp trick (computationally stable): subtract maxA0 before exponentiating
expA0t=exp(bsxfun(@minus,A0,maxA0));
% total (un-normalized) likelihood at each voxel -> pre-divided by maxA0
sum_expA0t=sum(expA0t,2);

% ===== [gamm_kv] ===== %
% responsibilities, computed by renormalizing so that sum over clusters = 1, for each voxel
x1=bsxfun(@rdivide,expA0t,sum_expA0t)';

% =====   [LLIK]  ===== %
% total model log-likelihood, computed by applying log to likelihoods, then re-adding maxA0 term
x2=sum(log(sum_expA0t)+maxA0);

%% [cluster priors]*
function x = MSTEP_pi_k( gamm_kv, dim )

% average responsibility over voxels, for each k
x = sum( gamm_kv, 2 ) ./ dim.V;

%% [means v1]
function x = MSTEP_mu_tks( X, sig2_kvs, gamm_kv, w_kv, dim  )

% initialize matrix of cluster means
x = zeros(dim.T,dim.K,dim.S);

for(s=1:dim.S)   
    % for each k, we have sum on vox
    vprodsum = sum( gamm_kv.*(w_kv.^2)./sig2_kvs(:,:,s), 2);
    % weighted-average timeseries, per subject and cluster
    x(:,:,s) = bsxfun(@rdivide, ((gamm_kv.*w_kv./sig2_kvs(:,:,s))*X(:,:,s))', vprodsum' );
end

%% [noise variance]*
function x = MSTEP_sig2_kvs( X, gamm_kv, mu_tks, w_kv, lambda, dim )

% initialize matrix of variance values
x = zeros(dim.K,dim.V,dim.S);

for(s=1:dim.S)
    % Get squared distance of [X - w_kv*(mu_k)] per voxel, in expanded form
    del=bsxfun(@minus,sum(X(:,:,s).^2,2)',2*w_kv.*(X(:,:,s)*mu_tks(:,:,s))')+bsxfun(@times,w_kv.^2,sum(mu_tks(:,:,s).^2)');    
    % compute weighted average of square-distance, divided by #timepoints

    if(dim.PRIOR==1)
    % improper - negative exponential distribution
    x(:,:,s) = ( (del .* gamm_kv) + lambda ) ./ ( (dim.T .* gamm_kv) + lambda );
    elseif(dim.PRIOR==2)
    % inverse gamma distribution
    x(:,:,s) = ( (del .* gamm_kv) + 2*lambda(2) ) ./ ( (dim.T .* gamm_kv) + 2*(lambda(1)+1) );
    else
    % no priori
    x(:,:,s) = del ./ dim.T;     
    end
end

%% [voxel scaling]*
function x = MSTEP_w_kv( X, mu_tks, sig2_kvs, dim )

for(s=1:dim.S)    
    if s==1
        aa_set2 = X(:,:,s)*mu_tks(:,:,s)./sig2_kvs(:,:,s)';             
        bb2     = bsxfun(@rdivide,squeeze(sum( mu_tks(:,:,s).^2,1)),sig2_kvs(:,:,s)');
    else
        aa_set2 = aa_set2+X(:,:,s)*mu_tks(:,:,s)./sig2_kvs(:,:,s)';     
        bb2    = bb2+bsxfun(@rdivide,squeeze(sum( mu_tks(:,:,s).^2,1)),sig2_kvs(:,:,s)');
    end
end
x = (aa_set2./bb2)';
% non-negativity constraint
x(x<=0) = 0;

%% [subject scaling]*
function x = MSTEP_c_ks( X, sig2_kvs, w_kv, gamm_kv, H, P_s, dim )

% initialize matrix of scaling values
x = zeros(dim.K,dim.S);

for(s=1:dim.S)
    % predefine product of voxel-weighting terms:
    vox_wt = gamm_kv .* w_kv ./ sig2_kvs(:,:,s);
    % numerator term: (sum of [voxel-weighting terms * x_sv]) x (P_s * h_k)
    aa = sum((vox_wt*X(:,:,s)).*(P_s(:,:,s)*H)',2);    
    % predefine product of voxel-weighting terms:
    vox_wt = gamm_kv .* (w_kv.^2) ./ sig2_kvs(:,:,s);
    % denominator term: (sum of voxel-weighting terms) x (inner product of h_k)
    bb = sum( vox_wt, 2 ) .* sum( H.^2, 1 )'; % (1xK) set of terms
    % ratio of terms for kth cluster
    x(:,s) = aa./bb;
end

% non-negativity constraint
x(x<=0) = 0;

% rescale to unit norm (turned off for now)
%x = bsxfun(@rdivide, x, sqrt(sum(x.^2,2)) );

%% [Ps orthonormal basis set  +  H fixed-covariance matrix]*
function [x1, x2] = MSTEP_P_s_and_H( X, c_ks, sig2_kvs, w_kv, gamm_kv, H, dim )

% common term in both P_s and H solutions: (k x t) matrix, summed on voxels (v) 
% here we index over subjects (s) on 3rd dimension
bb_set = zeros(dim.K,dim.T,dim.S);
%
for(s=1:dim.S)    
    % for efficiency, we compute sum on (v), for each cluster k=1...K:    
    bb_set(:,:,s) = (gamm_kv .* w_kv ./ sig2_kvs(:,:,s)) * X(:,:,s);
end

% ===== computing P_s ===== %

% initialize P_s, 3D matrix of timeseries
x1 = zeros(dim.T,dim.K,dim.S);
% other term:  product of matrices indexed only by subject (s)
% get 3D matrix containing all subject terms (stacked on 3rd dim)
aa_set  = bsxfun(@times, H, permute( c_ks, [3 1 2] ) );
%
for(s=1:dim.S)
    % take the SVD of product each subject (aa*bb) matrix
    [Ux Lx Vx] = svd( aa_set(:,:,s)*bb_set(:,:,s), 'econ' );
    % P_s is the product of left/right orthonormal bases
    x1(:,:,s) = Vx*Ux';
end

% ===== computing H ===== %

% first term: diagonal matrices, computed element-wise products
% includes terms that are independent sums on (v), and (s):
aa_all = sum( gamm_kv .* w_kv.^2 .* permute( sum( bsxfun(@rdivide, c_ks.^2, permute(sig2_kvs,[1 3 2])), 2), [1 3 2] ), 2 );

% second term: contains interdependent sums on (s,v)
% NB: we compute the transpose of (small) cc matrix, so that we needn't transpose larger matrix X
cc = zeros(dim.K,dim.K); %% = outer sum on (s)

for(s=1:dim.S)
    % add to sum over subjects
    cc = cc + diag( c_ks(:,s) ) * bb_set(:,:,s) * x1(:,:,s);
end

% H given by product of these matrices
x2 = cc'*diag(1./aa_all);

%% ======================================================= %%
%%  EFFICIENT IMPLEMENTATION OF K-MEANS FOR INITIALIZATION %%
%% ======================================================= %%

function [ out ] = kmeans_group( Xdat, K, NLOOP )
%
% KMEANS CLUSTERING MODEL:
%
% for efficient solution in 3D (v x t x s) data matrix.
% Chooses the max-likelihood solution over NLOOP iterations
%

cbest = Inf; %% initialize cost function

for(bitloop=1:NLOOP) %% outer loop ... random-start initializations

    disp(['kmeans ',num2str(bitloop),' of ',num2str(NLOOP)]);
    
    [V T S] = size(Xdat);  % size of data matrix
    LAB     = zeros(V,1);  % label list
    LAB_old = zeros(V,1);  % "old" label list
    
    % initialize by random selection of K regions
    % maxtrix of cluster means has dimensions: mu = (K x T x S)
    list   = randperm( V );
    mu_kts = Xdat(list(1:K),:,:); % K random seeds
    dist   = zeros( V,K ); % initialized Euclid. distance matrix
    
    change=1; %% fraction of voxels where cluster assign. changes
    iter  =0; %% number of iterations        
    
    %% terminate if <5% change or >20 iterations
    while( (change>0.05) && (iter<20) ) 
        
        iter=iter+1; %% increment iterations

%% ==== E-step ============================================================

        % compute Euclidean distance of voxels from cluster means        
        for(k=1:K)
            % average sq-distance over all subjects
            dist(:,k) = mean(sum( bsxfun(@minus, Xdat, mu_kts(k,:,:)).^2, 2),3);            
        end        
        % allocate voxel to most probable group
        [v LAB] = min( dist,[],2 );

%% ==== M-step ============================================================

        % compute cluster means from new partitions
        for(k=1:K)
            %
            mu_kts(k,:,:) = mean( Xdat(LAB==k,:,:),1 );
        end
        
        % fraction of voxels that change assignment
        change  = 1 - sum( double(LAB_old==LAB) )/V; 
        LAB_old = LAB; % update "old" label-set
    end
    
    % sum of sq distance, averaged over subjects, rel. cluster centroids
    cdist = zeros(1,1);
    for(k=1:K)
       cdist = cdist + mean(sum(sum(  bsxfun(@minus, Xdat(LAB==k,:,:), mu_kts(k,:,:)).^2, 2 ),1),3);
    end
    
    % check if sol'n better than current optimum
    if( cdist < cbest ) 
        % if true, current "best" error:
        cbest      = cdist;
        % store results
        out.LAB    = LAB;
        out.mu_kts = mu_kts;
    end    
end

