function [r,lam,imat,oly] = gen_esd( X, alpha )
%
% . does generalized extreme studentized deviate (ESD) test of
% . Rosner (1983), which identifies multiple outliers in an *approximately*
% . normal distribution
%

[n,k] = size(X);
imax  = round( 0.2*n ); % up to 20% potential outliers
ilist = (1:imax)';
plist = 1 - alpha./(2*(n-ilist+1));
tlist = tinv( plist, n-ilist-1 );

lam = (n-ilist).*tlist ./ sqrt( (n-ilist+tlist.^2).*(n-ilist+1) );

for j=1:k
    xtmp = X(:,j);
    itmp = (1:n)';
    for i=1:imax
        % largest entry, w max-abs deviation from mean
        [vx,ix] = max( abs(xtmp-mean(xtmp)) ); 
        % test statistic on largest entry
        r(i,j) = vx/std(xtmp);
        % store index of largest entry
        imat(i,j) = itmp(ix);
        % delete largest entry
        xtmp(ix)=[];
        itmp(ix)=[];
        %
        if r(i,j)>lam(i)
            oly(i,j)=1;
        else
            oly(i,j)=0;
        end
    end
end
