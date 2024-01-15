function Z_est = imputation_models( X, str )

if(str.submean>0)
    Favs = mean(X, 1,'omitnan'); % variable means
    X    = bsxfun(@minus,X,Favs);
end
if(str.subvar>0)
    Fsds = std(X,0,1,'omitnan'); % variable means
    X    = bsxfun(@rdivide,X,Fsds);
end

Xin = X;
P   = isfinite(Xin);
Xin(~isfinite(Xin))=0;

if( strcmpi(str.type,'mean') )
    
    avs = sum(Xin.*P)./sum(P); % variable means
    Z_est = Xin.*P + bsxfun(@times,(1-P), avs);

elseif( strcmpi(str.type,'UVN') )
    
    for(p=1:size(X,2))
       [m,s]=ecmnmle(X(:,p)); 
       sim(:,p) = (randn(size(X,1),1) .* sqrt(s)) + m;
    end
    Z_est = Xin.*P + (1-P).*sim;
    
elseif( strcmpi(str.type,'MVN') )
    
    [m,S] = ecmnmle(X);
    Z_est  = Xin;
    for(n=1:size(X,1))
       xtmp = X(n,:)';
       if(sum(isfinite(xtmp))==0) % if all missing
          xtmp = mvnrnd(m,S,1);
       elseif(sum(~isfinite(xtmp))>0) % if a subset missing
          xix = find(~isfinite(xtmp));
          yix = find( isfinite(xtmp));
          mt  = m(xix,1) + S(xix,yix)*inv(S(yix,yix))*(xtmp(yix,1)-m(yix,1));
          St  = S(xix,xix) - S(xix,yix)*inv(S(yix,yix))*S(yix,xix);
          xtmp(xix) = mvnrnd(mt,St,1);
       end
       Z_est(n,:) = xtmp; 
    end
end

if(str.subvar>0)
    Z_est= bsxfun(@times,Z_est,Fsds);
end
if(str.submean>0)
    Z_est= bsxfun(@plus,Z_est,Favs);
end

