function LD = LD_NR_train (data, labels, Range)

% data in full basis
data1 = data (:, labels == 1);
data0 = data (:, labels == 0);
% mean for all bases
mean_dist    = mean (data1, 2) - mean (data0, 2);
% initialize
lin_discr = zeros( Range, Range );

for(k=1:Range)

    within_class     = cov (data1(1:k,:)') + cov (data0(1:k,:)');
    lin_discr(1:k,k) = inv (within_class) * mean_dist(1:k);
end

LD.lin_discr = lin_discr;
