function out = gamthr( X, alpha, comb )
%
% GAMTHR: simple script for executing gamm distribution fit + thresholding
% of strictly positive data, for identification of outliers, etc.
%
% syntax: 
%           out = gamthr( X, alpha, comb )
% input:
%           X     : data matrix (N samples x P variables)
%           alpha : nominal significance level
%           comb  : method for combining stats when tresholding (in case of P>1)
%                   either 'union'     (signif if ANY p<alpha)
%                   or     'intersect' (signif if ALL p<alpha)

% data dimensions
[N,P] = size(X);
% make sure values nonnegative, rescale to [0,1] bounded
X(X<eps)=eps;
X=bsxfun(@rdivide,X,max(X,[],1));

for(p=1:P)
   % p-value estimation
   [parmhat] = gamfit( X(:,p) );
   out.pp(:,p)   = 1-gamcdf( X(:,p),parmhat(1),parmhat(2) );
end

out.thr      = double( out.pp<= alpha);

if( P>1 )
    if    (strcmpi(comb,'intersect')) %%  all must be <pthr
        out.pp_comb = max(out.pp,[],2);
    elseif(strcmpi(comb,'union'))     %%  at least 1 must be <pthr
        out.pp_comb = min(out.pp,[],2);
    end
    out.thr_comb = double( out.pp_comb<= alpha);
end