function out = bootreg( X, Y )
%
% DIRECTION IST FLIPPEDED
%
% for reg, quickboot( [DV (outcome)], [IV (predictor)], 'reg' )

NITER=2000;


fin = isfinite( X ) & isfinite( sum(Y,2) );
X=X(fin);   % outcome (DV)
Y=Y(fin,:); % predictor(s) (IV)

for(bsr=1:NITER)
    list = ceil( length(X)*rand(length(X),1) );
    xbs=X(list);
    ybs=[ones(numel(list),1), Y(list,:)];
    btmp = xbs' * (ybs / (ybs'*ybs));
    beta(bsr,:) = btmp;
end

xbs=X(:);
ybs=[ones(numel(list),1), Y];
btmp = xbs' * (ybs / (ybs'*ybs));

out.av = btmp;
out.se = std(beta,0,1);
out.br = out.av./out.se;
out.ci = prctile(beta,[2.5 97.5],1);
out.pp = 2*min([mean(beta<0,1); mean(beta>0,1)],[],1);
