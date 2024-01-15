function [accuracy] = LR_classify_set (LR, basis, train_means, test_data, test_labels);

% linear comb of basis vectors [vect vect ....]
dxSet = basis * LR.lin_discr;
% ...and projet: each row-vect = set of score-weihts
PROJ = dxSet' * test_data;
%
match    = (PROJ > 0) == repmat( test_labels(:)', [size(PROJ,1) 1] );
% fractional accuracy
accuracy = sum(match,2) ./ size(test_data,2);
