function out = infocalc( A, B, D )

A=zscore(A);
B=zscore(B);
D=zscore(D);

pAB=corr(A,B);
pAD=corr(A,D);
pBD=corr(B,D);
% marginal entropies
out.hA = 0.5*log( 2*pi*exp(1) );
out.hB = 0.5*log( 2*pi*exp(1) );
out.hD = 0.5*log( 2*pi*exp(1) );
% joint entropies
out.hAB  = 0.5*( 2*log( 2*pi*exp(1) ) + log(1-pAB^2) );
out.hAD  = 0.5*( 2*log( 2*pi*exp(1) ) + log(1-pAD^2) );
out.hBD  = 0.5*( 2*log( 2*pi*exp(1) ) + log(1-pBD^2) );
out.hABD = 0.5*( 3*log( 2*pi*exp(1) ) + log(1 + 2*pAB*pAD*pBD - pAB^2 - pAD^2 - pBD^2) );

% mutual information
% . "shared information" (or statistical dependency) between variable pairs
% .always >=0, with 0=statistically independent
out.iAB = out.hA + out.hB - out.hAB;
out.iAD = out.hA + out.hD - out.hAD;
out.iBD = out.hB + out.hD - out.hBD;
% conditional MI
% . "shared information" (or statistical dependency) between variable pairs
%   given a known third variable -- asserts conditional dependence relationships
out.iAB_D = (out.hAD - out.hD) + (out.hBD - out.hD) - (out.hABD - out.hD);
out.iAD_B = (out.hAB - out.hB) + (out.hBD - out.hB) - (out.hABD - out.hB);
out.iBD_A = (out.hAB - out.hA) + (out.hAD - out.hA) - (out.hABD - out.hA);
% 3way MI
% . shared information unique to 3-way group 
% . equivalently, measures information gain of adding a 3rd variable
%   (redundancy vs. synergy)
out.iABD  = -(out.hA + out.hB + out.hD) + (out.hAB + out.hAD + out.hBD) - out.hABD;

% total correlation information
% .full sum of depencence relationships between variables
out.TCI   = (out.hA+out.hB+out.hD) - out.hABD;
% conditional correlation information
% .total correlation between two variables given a known third
% .how much does 3rd variable explain correlations between first two
% .0=3rd contains full information
out.cAB_D = (out.iAD+out.iBD) - out.iABD;
