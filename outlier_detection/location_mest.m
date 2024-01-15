function ms = location_mest( X, k, Niters )
%
% LOCATION_MEST: robust estimator of location (Huber M-statistic)
%
% syntax:
%           ms = location_mest( X, k, Niters )
%
% input:
%           X       : input data matrices (P variables x N samples)
%           k       : normative tuning parameter (usually between 1.35 and 1.5)
%                     as k->Inf, statistic becomes less robust
%           Niters  : maximum number of iterations (usually Niters=30 is fine)
%


[P,N]=size(X);
%%% 1.345

alpha = 2*normcdf(k,0,1)-1;

m0 = median(X,2);
s0 = 1.49*median( abs(bsxfun(@minus,X,median(X,2))),2); %% if n<10 estimate degrades
s0(s0<eps)=eps;
ms = m0; 
mold = m0;

terminflag=0; iter=0;
while(iter<=Niters && terminflag==0)
    iter=iter+1;
    z = bsxfun(@rdivide,bsxfun(@minus,X,ms),s0);
    z(z<-k)=-k;
    z(z> k)= k;
    ms = ms + (s0.*sum(z,2))./(N*alpha);

    if( iter>3 && mean(abs(mold-ms))/mean(abs(mold)) < 2*eps ) 
        terminflag=1;
    else
        mold = ms;
    end
end
