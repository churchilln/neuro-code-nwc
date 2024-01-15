function out = quickboot( X, Y, type )
%
% .this is for quickly getting distros on a single variable
%  either 1-sample or 2-sample distribution, or associations with
%  a single other variable of interest, via correlation or regression
%
% for reg, quickboot( [DV (outcome)], [IV (predictor)], 'reg' )
%

if(nargin<2) Y=[]; end
if(nargin<3) type=''; end

NITER=2000;

if( isempty(Y) && (isempty(type) || strcmpi(type,'avg')) )

    X=X(isfinite(X));

    for(bsr=1:NITER)
        list = ceil( length(X)*rand(length(X),1) );
        bbsr(bsr,1) = mean(X(list));
    end

    out.av = mean(X(:));
    out.se = std(bbsr,'omitnan');
    out.ci = prctile(bbsr,[2.5 97.5]);
    out.pp = 2*min([mean(bbsr<0), mean(bbsr>0)]);

elseif( isempty(Y) && strcmpi(type,'med') )
    
    X=X(isfinite(X));

    for(bsr=1:NITER)
        list = ceil( length(X)*rand(length(X),1) );
        bbsr(bsr,1) = median(X(list));
    end

    out.av = median(X(:));
    out.se = std(bbsr,'omitnan');
    out.ci = prctile(bbsr,[2.5 97.5]);
    out.pp = 2*min([mean(bbsr<0), mean(bbsr>0)]);

elseif ( isempty(Y) && (strcmpi(type,'avg+') || strcmpi(type,'avg-')) )

    X=X(isfinite(X));

    for(bsr=1:NITER)
        list = ceil( length(X)*rand(length(X),1) );
        bbsr(bsr,1) = mean(X(list));
    end

    out.av = mean(X(:));
    out.se = std(bbsr,'omitnan');

    if strcmpi(type,'avg+') % test of presumed positive effect
        out.ci = prctile(bbsr,5)*[1 NaN];
        out.pp = mean(bbsr<0);
    elseif strcmpi(type,'avg-') % test of presumed negative effect
        out.ci = prctile(bbsr,95) * [NaN 1];
        out.pp = mean(bbsr>0);
    end

elseif ( isempty(Y) && (strcmpi(type,'med+') || strcmpi(type,'med-')) )

    X=X(isfinite(X));

    for(bsr=1:NITER)
        list = ceil( length(X)*rand(length(X),1) );
        bbsr(bsr,1) = median(X(list));
    end

    out.av = median(X(:));
    out.se = std(bbsr,'omitnan');

    if strcmpi(type,'med+') % test of presumed positive effect
        out.ci = prctile(bbsr,5)*[1 NaN];
        out.pp = mean(bbsr<0);
    elseif strcmpi(type,'med-') % test of presumed negative effect
        out.ci = prctile(bbsr,95) * [NaN 1];
        out.pp = mean(bbsr>0);
    end

else
    if(strcmpi(type,'corr_pe'))

        %fin = isfinite( X ) & isfinite( Y );
        %X=X(fin);
        %Y=Y(fin);

        for(bsr=1:NITER)
            list = ceil( length(X)*rand(length(X),1) );
            bbsr(bsr,1) = corr(X(list),Y(list),'rows','pairwise','type','Pearson');
        end

        out.av = corr(X(:),Y(:),'rows','pairwise','type','Pearson');
        out.se = std(bbsr,'omitnan');
        out.ci = prctile(bbsr,[2.5 97.5]);
        out.pp = 2*min([mean(bbsr<0), mean(bbsr>0)]);

    elseif(strcmpi(type,'corr_sp'))

        %fin = isfinite( X ) & isfinite( Y );
        %X=X(fin);
        %Y=Y(fin);

        for(bsr=1:NITER)
            list = ceil( length(X)*rand(length(X),1) );
            bbsr(bsr,1) = corr(X(list),Y(list),'rows','pairwise','type','Spearman');
        end

        out.av = corr(X(:),Y(:),'rows','pairwise','type','Spearman');
        out.se = std(bbsr,'omitnan');
        out.ci = prctile(bbsr,[2.5 97.5]);
        out.pp = 2*min([mean(bbsr<0), mean(bbsr>0)]);

    elseif(strcmpi(type,'pcorr_pe'))

        %fin = isfinite( X ) & isfinite( Y );
        %X=X(fin);
        %Y=Y(fin);

        for(bsr=1:NITER)
            list = ceil( length(X)*rand(length(X),1) );
             bbsr(bsr,1) = partialcorr(X(list),Y(list,1),Y(list,2:end),'rows','pairwise','type','Pearson');
        end

        out.av = partialcorr(X(:),Y(:,1),Y(:,2:end),'rows','pairwise','type','Pearson');
        out.se = std(bbsr,'omitnan');
        out.ci = prctile(bbsr,[2.5 97.5]);
        out.pp = 2*min([mean(bbsr<0), mean(bbsr>0)]);

    elseif(strcmpi(type,'pcorr_sp'))

        %fin = isfinite( X ) & isfinite( Y );
        %X=X(fin);
        %Y=Y(fin);

        for(bsr=1:NITER)
            list = ceil( length(X)*rand(length(X),1) );
             bbsr(bsr,1) = partialcorr(X(list),Y(list,1),Y(list,2:end),'rows','pairwise','type','Spearman');
        end

        out.av = partialcorr(X(:),Y(:,1),Y(:,2:end),'rows','pairwise','type','Spearman');
        out.se = std(bbsr,'omitnan');
        out.ci = prctile(bbsr,[2.5 97.5]);
        out.pp = 2*min([mean(bbsr<0), mean(bbsr>0)]);

    elseif( strcmpi(type,'R2'))
        
        fin = isfinite( X ) & isfinite( Y );
        X=X(fin);
        Y=Y(fin);

        for(bsr=1:NITER)
            list = ceil( length(X)*rand(length(X),1) );
            bbsr(bsr,1) = corr(X(list),Y(list)).^2;
        end

        out.av = corr(X(:),Y(:)).^2;
        out.se = std(bbsr,'omitnan');
        out.ci = prctile(bbsr,[2.5 97.5]);
        
    elseif( strcmpi(type,'reg'))
        

        fin = isfinite( X ) & isfinite( sum(Y,2) );
        X=X(fin);   % outcome (DV)
        Y=Y(fin,:); % predictor(s) (IV)

        for(bsr=1:NITER)
            list = ceil( length(X)*rand(length(X),1) );
            xbs=X(list);
            ybs=[ones(numel(list),1), Y(list,:)];
            btmp = xbs' * (ybs / (ybs'*ybs));
            bbsr(bsr,1) = btmp(2);
        end

        xbs=X(:);
        ybs=[ones(numel(list),1), Y];
        btmp = xbs' * (ybs / (ybs'*ybs));
        out.av = btmp(2);
        out.intercept = btmp(1);
        
        out.se = std(bbsr,'omitnan');
        out.ci = prctile(bbsr,[2.5 97.5]);
        out.pp = 2*min(cat(3,mean(bbsr<0), mean(bbsr>0)),[],3);

    elseif( strcmpi(type,'diff'))

        finx = isfinite( X );
        finy = isfinite( Y );
        X=X(finx);
        Y=Y(finy);
        
        for(bsr=1:NITER)
            listx = ceil( length(X)*rand(length(X),1) );
            listy = ceil( length(Y)*rand(length(Y),1) );
            bbsr(bsr,1) = mean(Y(listy))-mean(X(listx));
        end

        out.av = mean(Y(:))-mean(X(:));
        out.se = std(bbsr,'omitnan');
        out.ci = prctile(bbsr,[2.5 97.5]);
        out.pp = 2*min([mean(bbsr<0), mean(bbsr>0)]);

    elseif( strcmpi(type,'diffmed'))

        finx = isfinite( X );
        finy = isfinite( Y );
        X=X(finx);
        Y=Y(finy);
        
        for(bsr=1:NITER)
            listx = ceil( length(X)*rand(length(X),1) );
            listy = ceil( length(Y)*rand(length(Y),1) );
            bbsr(bsr,1) = median(Y(listy))-median(X(listx));
        end

        out.av = median(Y(:))-median(X(:));
        out.se = std(bbsr,'omitnan');
        out.ci = prctile(bbsr,[2.5 97.5]);
        out.pp = 2*min([mean(bbsr<0), mean(bbsr>0)]);
    
    elseif( strcmpi(type,'diff+') || strcmpi(type,'diff-') )

        finx = isfinite( X );
        finy = isfinite( Y );
        X=X(finx);
        Y=Y(finy);
    
        for(bsr=1:NITER)
            listx = ceil( length(X)*rand(length(X),1) );
            listy = ceil( length(Y)*rand(length(Y),1) );
            bbsr(bsr,1) = mean(Y(listy))-mean(X(listx));
        end

        out.av = mean(Y(:))-mean(X(:));
        out.se = std(bbsr,'omitnan');
        
        if strcmpi(type,'diff+') % test of presumed positive effect
            out.ci = prctile(bbsr,5)*[1 NaN];
            out.pp = mean(bbsr<0);
        elseif strcmpi(type,'diff-') % test of presumed negative effect
            out.ci = prctile(bbsr,95) * [NaN 1];
            out.pp = mean(bbsr>0);
        end

    elseif( strcmpi(type,'diffmed+') || strcmpi(type,'diffmed-') )

        finx = isfinite( X );
        finy = isfinite( Y );
        X=X(finx);
        Y=Y(finy);
    
        for(bsr=1:NITER)
            listx = ceil( length(X)*rand(length(X),1) );
            listy = ceil( length(Y)*rand(length(Y),1) );
            bbsr(bsr,1) = median(Y(listy))-median(X(listx));
        end

        out.av = median(Y(:))-median(X(:));
        out.se = std(bbsr,'omitnan');
        
        if strcmpi(type,'diffmed+') % test of presumed positive effect
            out.ci = prctile(bbsr,5)*[1 NaN];
            out.pp = mean(bbsr<0);
        elseif strcmpi(type,'diffmed-') % test of presumed negative effect
            out.ci = prctile(bbsr,95) * [NaN 1];
            out.pp = mean(bbsr>0);
        end
        
    elseif( strcmpi(type,'adjdiff') )
        
        fin = isfinite( X ) & isfinite( sum(Y,2) );
        X=X(fin);   % outcome (DV)
        Y=Y(fin,:); % predictor(s) (IV)

        % now splitting -- stratify on key variable
        dx  = Y(:,1); % pull out first reg
        X0  = X(dx==0);
        Y0  = Y(dx==0,:);
        X1  = X(dx==1);
        Y1  = Y(dx==1,:);
        
        for(bsr=1:NITER)
            list0 = ceil( length(X0)*rand(length(X0),1) );
            list1 = ceil( length(X1)*rand(length(X1),1) );
            
            xbs=[X0(list0); X1(list1)];
            ybs=[ones(numel(list0)+numel(list1),1), [Y0(list0,:);Y1(list1,:)]];
            btmp = xbs' * (ybs / (ybs'*ybs));
            bbsr(bsr,1) = btmp(2);
        end

        xbs=X(:);
        ybs=[ones(numel(list0)+numel(list1),1), [Y0;Y1]];
        btmp = xbs' * (ybs / (ybs'*ybs));
        out.av = btmp(2);
        out.intercept = btmp(1);
        
        out.se = std(bbsr,'omitnan');
        out.ci = prctile(bbsr,[2.5 97.5]);
        out.pp = 2*min(cat(3,mean(bbsr<0), mean(bbsr>0)),[],3);
        
        
    elseif( strcmpi(type,'diffpair'))

        finxy = isfinite( X ) & isfinite( Y );
        X=X(finxy);
        Y=Y(finxy);
        
%         nx=numel(X);
%         ny=numel(Y);
%         tmp = tiedrank([X;Y]);
%         X = tmp(1:nx);
%         Y = tmp(nx+1:end);

        for(bsr=1:NITER)
            listxy = ceil( length(X)*rand(length(X),1) );
            bbsr(bsr,1) = mean(Y(listxy))-mean(X(listxy));
        end

        out.av = mean(Y(:))-mean(X(:));
        out.se = std(bbsr,'omitnan');
        out.ci = prctile(bbsr,[2.5 97.5]);
        out.pp = 2*min([mean(bbsr<0), mean(bbsr>0)]);

    else
       error('unrecognized type option'); 
    end
end

if mean(~isfinite(bbsr))>0
    if mean(~isfinite(bbsr))>0.05
        error('too many non-finite bootstrap replicates! check your inputs for missing data!');
    else
        warning('some (<5%) non-finite bootstrap replicates! should be ok, but make sure you expected this!')
    end
end

disp('bootstrapping done!')
