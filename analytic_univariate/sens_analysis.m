function out = sens_analysis( Ymat, Xfix, Xvar )

[Nv,Nn] = size(Ymat);
[Nn,Nf] = size(Xfix);
[Nn,Nv] = size(Xvar);

% standardized everything...

Ymat = Ymat - mean(Ymat,2);
Ymat = Ymat ./ sqrt(sum(Ymat.^2,2));
Xfix = Xfix - mean(Xfix,1);
Xfix = Xfix ./ sqrt(sum(Xfix.^2,1));
Xvar = Xvar - mean(Xvar,1);
Xvar = Xvar ./ sqrt(sum(Xvar.^2,1));

% AIC
vidx = NaN*ones(2^Nv,Nv);
labmat   = zeros(2^Nv,Nv);
%
kq       = 1;
for(t=1:Nv)
   vmat = nchoosek(1:Nv,t); % matrix of combos
   for(u=1:size(vmat,1))
        kq=kq+1;
        vidx(kq,1:numel(vmat(u,:))) = vmat(u,:);
        for(k=1:Nv) labmat(kq,k) = sum( vmat(u,:)==k )>0; end
   end
end
Nprm = size(vidx,1);

% get the AIC model scores
for(t=1:Nprm)
    idx  = vidx(t, isfinite(vidx(t,:)) );
    Xtmp = [ones(Nn,1), Xfix, Xvar(:, idx )];
    Btmp = Ymat * Xtmp /( Xtmp'*Xtmp );
    % array of fixed values
    Bfix(:,:,t) = Btmp(:,(1:Nf)+1);
end

out.Bav = mean(Bfix,3);
out.Bsd = std(Bfix,0,3);

out.Bfix = Bfix;
out.vidx = vidx;