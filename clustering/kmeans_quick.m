function [ out ] = kmeans_quick( Xdat, K, NLOOP )
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

        % ==== catch in case of vanishing cluster, re-initialize ====
        if( length(unique(LAB)) < K )
            disp('k-means cluster collapsed! rebooting...');
            
            % initialize by random selection of K regions
            % maxtrix of cluster means has dimensions: mu = (K x T x S)
            list   = randperm( V );
            mu_kts = Xdat(list(1:K),:,:); % K random seeds
            dist   = zeros( V,K ); % initialized Euclid. distance matrix

            change=1; %% fraction of voxels where cluster assign. changes
            iter  =0; %% number of iterations        
        end
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
