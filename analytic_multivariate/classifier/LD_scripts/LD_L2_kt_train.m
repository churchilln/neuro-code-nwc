function LD = LD_L2_train (data, labels, Range)

[nv,nt]=size(data);

y = [labels(:), 1-labels(:)];
y = bsxfun(@rdivide,y,sum(y));
y = y(:,1)-y(:,2);
y = y/norm(y);

if( size(data,1)> 2*size(data,2) )
    
    W = data'*data;
    [~,s,~] = svd( W ); Lmax  = 2*trace(s);
    LAMBDA = exp(linspace( -log(Lmax), log(Lmax), Range )); %
    lin_discr = zeros( nv, length(LAMBDA) );
    warning off;
        for( q=1:length(LAMBDA) )
            %
            lin_discr(:,q) = (data / (  ( W   +   LAMBDA(q)*eye(nt) )  )) * y;
        end
    warning on;    
else
    W = data*data';
    [~,s,~] = svd( W ); Lmax  = 2*trace(s);
    LAMBDA = exp(linspace( -log(Lmax), log(Lmax), Range )); %
    lin_discr = zeros( nv, length(LAMBDA) );

    warning off;
        for( q=1:length(LAMBDA) )
            %
            lin_discr(:,q) = (  ( W   +   LAMBDA(q)*eye(nv) )  ) \ (data*y);
        end
    warning on;      
end

LD.lin_discr = lin_discr;
