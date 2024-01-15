function [U,Ladj,V,vrat] = SVD_nobias( X )

[U,L0,V]=svd( X,'econ' );
Q0 = L0*V';

Ladj = 0;
for(j=1:size(X,2))
    Qj     = Q0;
    Qj(:,j)= [];
    [Bj,~,~]=svd( Qj,'econ');
    zj   = Bj*(Bj'*Q0(:,j));
    Ladj = Ladj + zj.^2;
end

vrat = diag(L0.^2)./Ladj; % ratio of variance
Ladj = diag(sqrt(Ladj)); % bring scale back down
