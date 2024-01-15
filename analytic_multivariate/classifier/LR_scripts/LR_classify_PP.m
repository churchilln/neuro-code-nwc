function [accuracy] = LR_classify_PP (LR, basis, train_means, test_data, test_labels);

% linear comb of basis vectors [vect vect ....]
dxSet = basis * LR.lin_discr;
% ...and projet: each row-vect = set of score-weihts
PROJ = dxSet' * test_data;

pp1n = 1./(1+ exp(-PROJ));
pp0n = 1 - pp1n;

accuracy = mean([pp0n(:, test_labels(:)==0 ) pp1n(:, test_labels(:)==1 )],2);
