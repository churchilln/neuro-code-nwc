function [ U_group Eigs, Z_group, r ] = gCCA_group( Xdat, Npc_subj, Npc_group, Nresamp, stratify )

if( nargin<4 ) Nresamp = 50; end
if( nargin<5 ) stratify=[]; end


%--> canonical variate = new "observed" set of scores
%--> eigenvect = loadings on original variables
[Nobs Nvars Nsubj] = size(Xdat);

if(~isempty(stratify) && (numel(stratify) ~= Nsubj) ) 
    error('stratification list doesnt match datapoints'); 
end

disp('feat. selection...')
% feature selection step
Uset=zeros( Nobs, Npc_subj, Nsubj );
% iterate through subjects
for(s=1:Nsubj)
    % within-subject pca
    [ux l] = svd( Xdat(:,:,s),'econ' );
    % concat into matrix
    Uset(:,:,s) = ux(:,1:Npc_subj);
end

disp('running full-data gcca...');
% now run gcca
[Aset_0,~] = perform_gCCA( Uset, Npc_group );
Aset_F = bsxfun(@rdivide,Aset_0(:,1:Npc_group),sqrt(sum(Aset_0(:,1:Npc_group).^2)));

spm_sum=0;
for(iter=1:Nresamp)
    [ iter iter ],
    
    if(isempty(stratify))
        list = randperm(Nsubj); 
        l1   = list( 1:floor(Nsubj/2) );
        l2   = list( floor(Nsubj/2)+1:end );
    else
        Nsubset = numel(unique(stratify)); 
        list    = randperm(Nsubset); 
        l1t     = list( 1:floor(Nsubset/2) );
        l2t     = list( floor(Nsubset/2)+1:end );
        
        l1=[]; for(j=1:length(l1t)) l1=[l1; find(stratify(:)==l1t(j))]; end
        l2=[]; for(j=1:length(l2t)) l2=[l2; find(stratify(:)==l2t(j))]; end
    end
   
    disp('running split-1 gcca...');
    % now run gcca
    [Aset_0,~] = perform_gCCA( Uset(:,:,l1), Npc_group );
    Aset_1 = bsxfun(@rdivide,Aset_0(:,1:Npc_group),sqrt(sum(Aset_0(:,1:Npc_group).^2)));
    disp('running split-2 gcca...');
    % now run gcca
    [Aset_0,~] = perform_gCCA( Uset(:,:,l2), Npc_group );
    Aset_2 = bsxfun(@rdivide,Aset_0(:,1:Npc_group),sqrt(sum(Aset_0(:,1:Npc_group).^2)));

    oo = mini_procrust(Aset_F,Aset_1,'corr'); Aset_1 = Aset_1(:,oo.index)*diag(oo.flip); 
    oo = mini_procrust(Aset_F,Aset_2,'corr'); Aset_2 = Aset_2(:,oo.index)*diag(oo.flip); 
    
    [r(:,iter) spm_tmp] = get_rSPM( Aset_1, Aset_2, 1 );
    spm_sum = spm_sum + spm_tmp;
end


Z_group = spm_sum./Nresamp;


disp('outputs...');
% select subset, renormalized
U_group    = Aset_F(:,1:Npc_group);
U_group    = bsxfun(@rdivide,U_group, sqrt(sum(U_group.^2)));

for(s=1:Nsubj)
    %
    Eigs(:,:,s) = Xdat(:,:,s)'*U_group;
end    
Eigs = bsxfun(@rdivide,Eigs, sqrt(sum(Eigs.^2)));

% %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Aset cc_av] = perform_gCCA( SET, toppc )

[Nobs numPC Nsubj] = size(SET);

CCmat = zeros( Nsubj*numPC, Nsubj*numPC );
DDmat = zeros( Nsubj*numPC, Nsubj*numPC );
a     = zeros( Nobs, Nsubj);
Aset  = zeros( Nobs, toppc );

for(i=1:Nsubj)
for(j=i:Nsubj)

    rng_row = (1:numPC) + numPC*(i-1);
    rng_col = (1:numPC) + numPC*(j-1);
    
    CC = SET(:,:,i)' * SET(:,:,j);

    CCmat(rng_row, rng_col) = CC;
    CCmat(rng_col, rng_row) = CC';
    
    if(i==j)
       DDmat(rng_row,rng_col) = CC; 
    end
end
end

MM = (DDmat\(CCmat-DDmat))./(Nsubj-1);
[uu ss] = svd(MM);

for(k=1:toppc)
    %
    ush = reshape( uu(:,k), numPC, Nsubj );
    for(q=1:Nsubj)
       a(:,q) = SET(:,:,q) * ush(:,q); 
    end
    a = bsxfun(@rdivide,a,sqrt(sum(a.^2)));
    Aset(:,k) = mean(a,2);
    cc(:,k)   = (sum(corr(a),2) - 1)./(size(a,2)-1);
end

cc_av = mean(cc)';
