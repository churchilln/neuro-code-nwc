function [ FstatFit ] = fmap_fit( datamat, regressors )
%
% This model fits a matrix of P regressors - regressors [t samples x P]
% to N different variables/models - datamat [t samplex x N]
%
%      FstatFit = fmap_fit( datamat, regressors )
%
% gives an Nx1 vector of F-statistics
%
%
% ------------------------------------------------------------------------%
% Author: Nathan Churchill, University of Toronto
%  email: nathan.churchill@rotman.baycrest.on.ca
% ------------------------------------------------------------------------%
% version history: March 15 2012
% ------------------------------------------------------------------------%


[Nsamp Nregr] = size( regressors );

% Stat-fitting and regression
X         = [ones(Nsamp,1) regressors]; % add in baseline regressor
BetaFits  = inv(X'*X)*X' * datamat;     % B = inv(X'X)*X'*y 
model_est = X * BetaFits;               % <y> = X*B
% regress from this run
regrmat = datamat - model_est;
% get SumSquareErr denom + numerator            
SSerrnum  = sum( (model_est - repmat( mean(model_est), [Nsamp 1]) ).^2 );
SSerrdnm  = sum( regrmat.^2 );  refNaN = double(SSerrdnm==0);
% correct for cases with no effect (to avoid NaN)
SSerrnum(refNaN==1) = 0.0; 
SSerrdnm(refNaN==1) = 1.0;
% compute the SumSquareErr
SSerrEst = SSerrnum ./ SSerrdnm;
% SS*(n-p)/(p-1), since p = Nhfrq-1 
FstatFit = SSerrEst(:) * (Nsamp - Nregr+1 )/Nregr;      
