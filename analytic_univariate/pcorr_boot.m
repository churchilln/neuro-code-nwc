function out = pcorr_boot( y, X, Xtry )
%
% > partial coeerlation, univariate response vector
% > assesses coefs for regressors set X and/or impact of Xtry on coefs of X

NITER=2000;

if nargin<3
    Xtry = [];
end

y = y(:);

fin = isfinite( y ) & isfinite( sum(X,2) );
y=y(fin);   % outcome (DV)
X=X(fin,:); % predictor(s) (IV)
if ~isempty(Xtry)
    Xtry=Xtry(fin,:);
end

for(bsr=1:NITER)
    list = ceil( length(y)*rand(length(y),1) );
    ybs=y(list);
    Xbs=[X(list,:)];
    beta(bsr,:) = partialcorr( ybs, Xbs(:,1), Xbs(:,2:end) );

    if ~isempty(Xtry)
        Xbs_aug = [Xbs, Xtry(list,:)];
        beta_aug = partialcorr( ybs, Xbs_aug(:,1), Xbs_aug(:,2:end) );
        beta_dif(bsr,:) = beta_aug - beta(bsr,:);
    end

end
out.av = mean(beta,1,'omitnan');        
out.se = std(beta,0,1,'omitnan');
out.ci = prctile(beta,[2.5 97.5],1);
out.pp = 2*min(cat(3,mean(beta<0,1), mean(beta>0,1)),[],3);

if ~isempty(Xtry)
    out.av_dif = mean(beta_dif,1,'omitnan');        
    out.se_dif = std(beta_dif,0,1,'omitnan');
    out.ci_dif = prctile(beta_dif,[2.5 97.5],1);
    out.pp_dif = 2*min(cat(3,mean(beta_dif<0,1), mean(beta_dif>0,1)),[],3);
end

out.beta = beta;