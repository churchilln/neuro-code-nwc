function X_est = imputation_svd( X, str )

if(str.submean>0)
    Favs = mean(X, 1,'omitnan'); % variable means
    X    = bsxfun(@minus,X,Favs);
end
if(str.subvar>0)
    Fsds = std(X,0,1,'omitnan'); % variable means
    X    = bsxfun(@rdivide,X,Fsds+eps);
end

Xin = X;
P   = isfinite(Xin);
Xin(~isfinite(Xin))=0;
[n,p]=size(Xin);
r    =rank(Xin);

% firstpass initialization...
if(str.randomize==1)
    for(p=1:size(X,2))
       m=mean(X(:,p)  ,1,'omitnan');
       s= std(X(:,p),0,1,'omitnan');
       sim(:,p) = (randn(size(X,1),1) .* sqrt(s)) + m;
    end
    Zold = Xin.*P + (1-P).*sim; % start with random normal draw
elseif(str.randomize==0)
    Zold=Xin; % start with zeros 
end

if(strcmpi(str.type,'soft'))

    if(isempty(str.Lam))
        [~,D,~]=svd(Xin,'econ');
        minl   =min(D(D>0))/100; % lower bound: 1/100th of the smallest non-singular eigval
        maxl   =D(1,1)   - minl; % upper bound: largest eigval - 1/100th of smallest
        str.Lam = logspace( log10(maxl), log10(minl), 100 );
    end
    
    for(l=1:numel(str.Lam))
        %[l l],
        terminflag=0;
        iter=0;
                
        while( terminflag==0 && iter<100 )
            iter=iter+1;
            [U,D,V]=svd( (Xin.*P) + (Zold.*(1-P)),'econ' );
            D=diag(diag(D) - str.Lam(l));
            D(D<0)=0;
            Znew = U*D*V';
            %
            dev = (sum( (Znew(:)-Zold(:)).^2 )/sum(Zold(:).^2));
            if( iter>2 && dev < 1E-6 )
                terminflag=1;
                %iter,
            end
            %devSet(iter,1) = dev;
            Zold = Znew;
        end
        %figure,bar(devSet);
        Z_est(:,:,l) = Znew;
    end

    X_est = bsxfun(@plus, bsxfun(@times,Z_est,(1-P)), (Xin.*P) );
    
elseif( strcmpi(str.type,'hard') ) % hard thresholding on eigenvalues
            
    for(l=1:r)
        terminflag=0;
        iter=0;
                
        while( terminflag==0 && iter<100 )
            iter=iter+1;
            [U,D,V]=svd( (Xin.*P) + (Zold.*(1-P)),'econ' );
            Znew = U(:,1:l)*D(1:l,1:l)*V(:,1:l)';
            %
            dev = (sum( (Znew(:)-Zold(:)).^2 )/sum(Zold(:).^2));
            if( iter>2 && dev < 1E-6 )
                terminflag=1;
            end
            %devSet(iter,1) = dev;
            Zold = Znew;
        end
        %figure,bar(devSet);
        Z_est(:,:,l) = Znew;
    end
         
    X_est = bsxfun(@plus, bsxfun(@times,Z_est,(1-P)), (Xin.*P) );
    
elseif( strcmpi(str.type,'mean') ) % simplest model as ref. point
    
    avs = sum(Xin.*P)./sum(P); % variable means
    X_est = Xin.*P + bsxfun(@times,(1-P), avs);
end

if(str.subvar>0)
    X_est= bsxfun(@times,X_est,Fsds);
end
if(str.submean>0)
    X_est= bsxfun(@plus,X_est,Favs);
end
