function [uset] = svd_tensprod( Xcell, Ycell, NF )

dtmp = size(Xcell{1});
ns=dtmp(1);
px=dtmp(2:end);
nd=numel(px);
py=size(Ycell{1},2);

% setting up the full product tensor
xprod = [];
for(g=1:numel(Xcell))
    % standardized
    xtmp = Xcell{g};
    ytmp = Ycell{g};

%     xtmp = bsxfun(@minus,Xcell{g},mean(Xcell{g},1));
%     xtmp = bsxfun(@rdivide,xtmp,sqrt(sum(xtmp.^2,1)));
%     ytmp = bsxfun(@minus,Ycell{g},mean(Ycell{g},1));
%     ytmp = bsxfun(@rdivide,ytmp,sqrt(sum(ytmp.^2,1)));
    % linear products
    tmpprod=[];
    for(h=1:py)
       tmpprod = cat(1, tmpprod, sum( bsxfun(@times,xtmp,ytmp(:,h)), 1));
    end
    
    % concatenate groups
    xprod = cat(nd+2,xprod, tmpprod);
end
% product tensor ( dy(1) x dx(1) x dx(2) x ... dx(nd) x dg )

if(size(xprod,1)==1)    uvlab(1) = 1;
else                    uvlab(1)=0;
end
if(size(xprod,nd+2)==1) uvlab(2) = 1;
else                    uvlab(2)=0;
end

xprod = squeeze( xprod ); % reduced tensor -- assumes no x to squeeze
dset  = size(xprod); % size of each dim
ndnu  = numel(dset); % number of dims (order)

% initialized
for(i=1:ndnu)
   uset{i} = randn(dset(i),NF); 
   uset{i} = bsxfun(@rdivide,uset{i},sqrt(sum(uset{i}.^2)));
end

iter=0;
conv=1;
ucatold = 1E6;
while( iter<200 && conv>1E-8 )
    iter=iter+1;
    % go through fitting cycle
    ucatnew=[];
    for(i=1:ndnu)
        xtmp = xprod;
        for j=1:ndnu
            if j~=i
                xtmp = sum( bsxfun(@times, xtmp, permute(uset{j}(:,1),circshift(1:ndnu,j-1)) ), j);
            end
        end
        xtmp = squeeze(xtmp);
        uset{i}(:,1) = xtmp ./ norm(xtmp);
        ucatnew=[ucatnew; uset{i}(:,1)];
    end    
    conv=max( abs(ucatnew-ucatold) );
    %ctr(iter,1)=conv;
    ucatold=ucatnew;
end
fprintf('\n\ttermin: iter=%u, conv=%f\n',iter,conv),
