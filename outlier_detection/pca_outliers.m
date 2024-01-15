function out=pca_outliers( X, kmax )
%
% PCA_OUTLIERS: script to assess influence of datapoints in multivariate
% principal component space, using classic robust pca measures
%
% syntax:
%           out=pca_outliers( X, kmax )
%
% input:
%           X    : input data matrices (P variables x N samples)
%           kmax : maximum pca dimensionality to explore (kmax <= min(N,P))
%
% output:
%           out.mahal     : mahalanobis distance
%           out.lev       : leverage
%           out.orth      : orthogonality
%           out.recon_err : reconstruction error (leave-one-out "orth")
%

X=bsxfun(@minus,X,mean(X,2));

for(k=1:kmax)
    k,
    out.mahal(:,k) = diag( (q(1:k,:)'/(q(1:k,:)*q(1:k,:)') )*q(1:k,:) );
    out.lev(:,k)   = sum( q(1:k,:).^2, 1 )./trace(l(1:k,1:k).^2);
    out.orth(:,k)  = sum( (X - (u(:,1:k)*l(1:k,1:k)*v(:,1:k)')).^2 );
end

for(t=1:size(X,2))
    t,
    Xtrn = X; Xtrn(:,t)=[];
    [u,l,v]=svd(Xtrn,'econ');
    scr = u(:,1:kmax)'*X(:,t);
    for(k=1:kmax)
    out.recon_err(t,k) = mean( (X(:,t) - (u(:,1:k)*scr(1:k))).^2 );
    end
end
