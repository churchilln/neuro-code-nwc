function out = variance_partition( Ymat, Xmat, aiccor )

if nargin<3
    aiccor=0;
end

[Nv,Nn] = size(Ymat);
[Nn,Nk] = size(Xmat);

% standardized everything...

Ymat = Ymat - mean(Ymat,2);
Ymat = Ymat ./ sqrt(sum(Ymat.^2,2));
Xmat = Xmat - mean(Xmat,1);
Xmat = Xmat ./ sqrt(sum(Xmat.^2,1));

% standardized coefficients
B = (Ymat * Xmat) /( Xmat'*Xmat );
% correlation coefficients
C = (Ymat * Xmat);
% coeff of determination
Yhat = B*Xmat';
R2   = 1 - sum( (Ymat-Yhat).^2,2 )./sum( Ymat.^2,2 );

% pratt index
PI = B .* C;

% relative weight index
[u,l,v]=svd( Xmat,'econ' );
z=u*v';
bx=Xmat'*z;
by=Ymat *z;
for(k=1:size(Xmat,2))
   RW(:,k) = sum( by.^2 .* bx(k,:).^2, 2);
end

% % generalized dominance analysis
% for(k=1:Nk)
%     xh = Xmat(:,k); % held
%     xr = Xmat; xr(:,k)=[]; % remain
%    
%     % indexing
%     vidx = NaN*ones(2^(Nk-1),Nk-1);
%     %
%     kq       = 1;
%     for(t=1:(Nk-1))
%        vmat = nchoosek(1:(Nk-1),t); % matrix of combos
%        for(u=1:size(vmat,1))
%             kq=kq+1;
%             vidx(kq,1:numel(vmat(u,:))) = vmat(u,:);
%             endvaria
%     end
%     Nprm = size(vidx,1);
%     
%     gtmp=0;
%     for(t=1:Nprm)
%         idx  = vidx(k,:);
%         % model without
%         Xtmp = [ones(Nn,1), xr(:, idx(isfinite(idx)) )];
%         Btmp = Ymat * Xtmp /( Xtmp'*Xtmp );
%         Yhat = Btmp*Xtmp';
%         R2o  = 1 - sum( (Ymat-Yhat).^2,2 )./sum( Ymat.^2,2 );
%         % model with
%         Xtmp = [ones(Nn,1), xh, xr(:, idx(isfinite(idx)) )];
%         Btmp = Ymat * Xtmp /( Xtmp'*Xtmp );
%         Yhat = Btmp*Xtmp';
%         R2i  = 1 - sum( (Ymat-Yhat).^2,2 )./sum( Ymat.^2,2 );
%         %
%         gtmp = gtmp + (R2i-R2o)./Nprm;
%     end
%     GD(:,k) = gtmp;
% end


% AIC
vidx = NaN*ones(2^Nk,Nk);
labmat   = zeros(2^Nk,Nk);
%
kq       = 1;
for(t=1:Nk)
   vmat = nchoosek(1:Nk,t); % matrix of combos
   for(u=1:size(vmat,1))
        kq=kq+1;
        vidx(kq,1:numel(vmat(u,:))) = vmat(u,:);
        for(k=1:Nk) labmat(kq,k) = sum( vmat(u,:)==k )>0; end
   end
end
Nprm = size(vidx,1);

% get the AIC model scores
for(t=1:Nprm)
    idx  = vidx(t, isfinite(vidx(t,:)) );
    Xtmp = [ones(Nn,1), Xmat(:, idx )];
    Btmp = Ymat * Xtmp /( Xtmp'*Xtmp );
    Yhat = Btmp*Xtmp';
    s2   = var( Ymat - Yhat,1,2);
    dif  = sum( (Ymat - Yhat).^2,2);
    LL   = -(Nn/2) .* (log(2*pi) + log(s2)) - (1./(2.*s2)).*dif;
    if aiccor==0
        AIC(:,t) = 2*(size(Xtmp,2)+1) - 2*LL;
    elseif aiccor==1
        KK = size(Xtmp,2)+1;
        AIC(:,t) = 2*KK - 2*LL + 2*KK*(KK+1)/(Nn-KK-1);
    end
end

ee = exp( -0.5* (AIC-min(AIC,[],2)) ); ee=ee./sum(ee,2);
er = max(ee,[],2)./ee;

% get the model averaging
Barr = zeros( Nv, Nk );
for(t=1:Nprm)
    idx  = vidx(t, isfinite(vidx(t,:)) );
    Xtmp = [ones(Nn,1), Xmat(:, idx )];
    Btmp = Ymat * Xtmp /( Xtmp'*Xtmp );
    Barr(:,idx) = Barr(:,idx) + bsxfun(@times, Btmp(:,2:end), ee(:,t));
end

for(k=1:Nk)
   AIP(:,k) = sum( ee(:,labmat(:,k)>0),2 );
end

%%

out.CC = C;
out.R2 = R2;
out.PI = PI;
out.RW = RW;
% out.GD = GD;
out.AIC = AIC;
out.AIP = AIP;
out.AIB = Barr;
out.vidx = vidx;