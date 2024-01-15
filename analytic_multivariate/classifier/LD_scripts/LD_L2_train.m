function LD = LD_L2_train (data, labels, Range)

% dataset
data1 = data (:, labels == 1);
data0 = data (:, labels == 0);
% linear discriminant
mean_dist    = mean (data1, 2) - mean (data0, 2);
within_class = cov (data1') + cov (data0');

[u s v] = svd( within_class ); Lmax  = 2*trace(s);
% declare range of regularization
LAMBDA = exp(linspace( -log(Lmax), log(Lmax), Range )); %
% initialize discriminant matrix
lin_discr = zeros( size(data1,1), length(LAMBDA) );
% test range
    
warning off;
    for( q=1:length(LAMBDA) )
        %
        lin_discr(:,q) = inv (  (within_class + LAMBDA(q)*eye(length(mean_dist)) )  ) * mean_dist;
    end
warning on;
    
% output
LD.lin_discr = lin_discr;
