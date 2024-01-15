function out = pcr_r2( X, y, Niter )

varcorr=0; 


Nvox = size(X,1);
N = size(X,2);
% on full data
X = bsxfun(@minus,X,mean(X,2));
y = zscore((y));
% svd -- data reduction
[U L V] = svd( X,'econ' );
Q0 = U'*X;
% 
Ntst = round(0.1*N);
if( Ntst< 2 ) Ntst = 2; end
Ntrn = N-Ntst;

n_ext = Niter*Ntst;
n_trn = N;

PRESS_tot = 0;
PRESS_set = zeros(Niter,1);
kmax = Ntrn-1;
pross=zeros(N,1);
prTot=zeros(N,1);
for(iter=1:Niter)

    list = randperm( N );
    
    % split data
    Qtrn = Q0(:,list(1:Ntrn)); 
    Qtst = Q0(:,list(Ntrn+1:end));
    ytrn = y(list(1:Ntrn));
    ytst = y(list(Ntrn+1:end));
    % centering
    Qtst   = bsxfun(@minus, Qtst,mean(Qtrn,2)); %subtract training mean
    Qtrn   = bsxfun(@minus, Qtrn,mean(Qtrn,2)); %subtract training mean
    ytst   = ytst - mean(ytrn); %subtract training mean
    ytrn   = ytrn - mean(ytrn); %subtract training mean
    % re-pca
    [u2 l2 v2] = svd(Qtrn,'econ'); u2=u2(:,1:kmax-5);
    
    %q2t=u2'*Qtrn; 
    if(varcorr==0)
        q2t = l2(1:kmax-5,1:kmax-5)*v2(:,1:kmax-5)';    
    else
        for(j=1:size(Qtrn,2))
            q2t_del =Qtrn;
            q2t_del(:,j)=[];
            q2t_avg=mean(q2t_del,2);
            [u_del l_del v_del] = svd( bsxfun(@minus,q2t_del,q2t_avg),'econ');
            z(:,j) = u_del'*(Qtrn(:,j)-q2t_avg);	
        end      
        whos;
        var_x2 = sum( z.^2, 2 );
        q2t    = diag(sqrt(var_x2))*v2(:,1:kmax);
    end
    
    
    q2h=u2'*Qtst; 
    % model fit
    w0 = pinv( (q2t*q2t'))*q2t*ytrn;
    y_hat = q2h'*w0;
    %qaug = [q2t; ones(1,Ntrn)]; w0 = pinv( (qaug*qaug'))*qaug*ytrn;    
    %qaug = [q2h; ones(1,Ntst)]; y_hat = qaug'*w0;
    PRESS_tot = PRESS_tot + sum( (y_hat - ytst).^2 );
    PRESS_set(iter) = sum( (y_hat - ytst).^2 );
    %
    pross( list(Ntrn+1:end) ) = pross( list(Ntrn+1:end) ) + y_hat;
    prTot( list(Ntrn+1:end) ) = prTot( list(Ntrn+1:end) ) + 1;
end
out.pross = pross./prTot;

TSS    = sum( (y-mean(y)).^2 );
out.r2 = 1 - ( (PRESS_tot/n_ext)/(TSS/n_trn) );

r2_b=zeros(5000,1);
for(bsr=1:5000)
   list = ceil( Niter*rand(Niter,1) );
   r2_b(bsr) = 1 - ( (sum(PRESS_set(list))/n_ext)/(TSS/n_trn) );
end
out.p = 1-mean( r2_b>0 );
out.r2_95 = prctile( r2_b, [2.5 97.5] );

%%
return;
if( out.p < 0.10 )

    for(bsr=1:1000)
        bsr,
        list = ceil( length(y)* rand( length(y),1 ) );
        % re-transform
        Q0t   = Q0(:,list);
        Q0tav = mean(Q0t,2);
        Q0t   = bsxfun(@minus, Q0t,Q0tav);
        [u2 l2 v2] = svd(Q0t,'econ'); u2=u2(:,1:kmax);
        q2t=u2'*Q0t; 
        %
        yt = y(list);
        yt = yt-mean(yt);
        w0 = pinv( (q2t*q2t') )*q2t*yt;
        map(:,bsr) = U*u2*w0;
    end

    if( Nvox <= 175000 )
    %% ==============================================================
        STEP=50000;
        for(v = 1:STEP:Nvox)
            list = v:v+STEP;
            list(list>Nvox)=[];
            out.map(list,1) = mean(map(list,:),2)./std(map(list,:),0,2);
        end
    %% ==============================================================
    else
        out.map = mean(map,2)./std(map,0,2);
    end
    map = bsxfun(@rdivide,map,sqrt(sum(map.^2)));


    if( Nvox <= 175000 )
    %% ==============================================================
        STEP=50000;
        for(v = 1:STEP:Nvox)
            list = v:v+STEP;
            list(list>Nvox)=[];
            out.mapN(list,1) = mean(map(list,:),2)./std(map(list,:),0,2);
        end
    %% ==============================================================
    else
        out.mapN = mean(map,2)./std(map,0,2);
    end
end
