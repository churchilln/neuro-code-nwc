function [accuracy] = LD_classify_set (LD, basis, train_means, test_data, test_labels);

% linear comb of basis vectors [vect vect ....]
dxSet = basis * LD.lin_discr;
% difference from mean in data-space
DIF_0 = test_data - repmat( train_means.mean0, [1 size(test_data,2)] );
DIF_1 = test_data - repmat( train_means.mean1, [1 size(test_data,2)] );
% basis x scan
log_pp0 = -0.5*(dxSet'*DIF_0).^2;
log_pp1 = -0.5*(dxSet'*DIF_1).^2;

% binary if prediction matches design
match    = double( double(log_pp1 > log_pp0) == repmat( test_labels(:)', [size(log_pp1,1) 1] ));
% fractional accuracy
accuracy = sum(match,2) ./ size(test_data,2);
