function [ U_group Eigs rr_0] = gCCA_group( Xdat, Npc_subj, Npc_group )

%--> canonical variate = new "observed" set of scores
%--> eigenvect = loadings on original variables
[Nobs Nvars Nsubj] = size(Xdat);

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
% Uset=bsxfun(@minus,Uset,mean(Uset,1));

disp('running gcca...');
% now run gcca
[Aset_0, rr_0] = perform_gCCA( Uset, Npc_group );
disp('outputs...');
% select subset, renormalized
U_group    = Aset_0(:,1:Npc_group);
U_group    = bsxfun(@rdivide,U_group, sqrt(sum(U_group.^2)) );

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
% [V,D]= eig((CCmat-DDmat)./(Nsubj-1),DDmat);

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
