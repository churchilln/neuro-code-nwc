function out = boot_corr( X, Yreg, Niter, ranknorm )
%
% bootstrapped correlations

[Nsamp Nvox] = size(X);
[Nsamp Nbeh] = size(Yreg);

if( Nbeh==1 && length(unique(Yreg))==2 )
%% matched-sample mean differences
    pset =zeros(Nbeh,Nvox);
    sprav=zeros(Nbeh,Nvox);
    
    if(ranknorm>1) X    = tiedrank(X);     end
    glav = mean(X); %% average map
    
    yvl = unique(Yreg);
    X1 = X( Yreg==yvl(1), : );
    X2 = X( Yreg==yvl(2), : );
    
    for(bsr=1:Niter)
        if( rem(bsr,50)==0 )
        [bsr Niter],
        end
        list1 = ceil( size(X1,1)*rand(size(X1,1),1) );
        list2 = ceil( size(X2,1)*rand(size(X2,1),1) );
        %%
        map  = mean( X2(list2,:) ) - mean( X1(list1,:) );
        pset = pset +double(map>0);
        sprav= sprav +map;
    end
    pset = pset./bsr;
    sprav=sprav./bsr;

    out.pmap = pset;
    out.spm  = sprav;  
    out.fract= sprav./glav;
else
%% unmatched correlation

    pset =zeros(Nbeh,Nvox);
    sprav=zeros(Nbeh,Nvox);
    
    if(ranknorm>0) Yreg = tiedrank(Yreg);  end
    if(ranknorm>1) X    = tiedrank(X);     end
    
    for(bsr=1:Niter)
        if( rem(bsr,50)==0 )
        [bsr Niter],
        end
        list = ceil( Nsamp*rand(Nsamp,1) );
        
        Yb   = zscore((Yreg(list,:)));        
        Xb   = zscore(X(list,:)   );
        map  = (Yb'*Xb)./(Nsamp-1);
        pset = pset +double(map>0);
        sprav= sprav +map;
    end
    pset = pset./bsr;
    sprav=sprav./bsr;

    out.pmap = pset;
    out.spm  = sprav;
end

% convert to 2-tailed
out.pmap = 2*min(cat(3,out.pmap,1-out.pmap),[],3);
