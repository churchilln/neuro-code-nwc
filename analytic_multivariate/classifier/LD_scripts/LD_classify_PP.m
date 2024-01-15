function [accuracy] = LD_classify_PP (LD, basis, train_means, test_data, test_labels);

% linear comb of basis vectors [vect vect ....]
dxSet = basis * LD.lin_discr;
% difference from mean in data-space
DIF_0 = test_data - repmat( train_means.mean0, [1 size(test_data,2)] );
DIF_1 = test_data - repmat( train_means.mean1, [1 size(test_data,2)] );

% basis x scan
pp0  = exp( -0.5*(dxSet'*DIF_0).^2 );
pp1  = exp( -0.5*(dxSet'*DIF_1).^2 );
pp0n = pp0./(pp0+pp1);
pp1n = pp1./(pp0+pp1);

pp0n( ~isfinite(pp0n) ) = 0.5;
pp1n( ~isfinite(pp1n) ) = 0.5;

% pprobers
accuracy = mean([pp0n(:, test_labels(:)==0 ) pp1n(:, test_labels(:)==1 )],2);
