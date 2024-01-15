function out = pcr_r2( X, y )

varcorr=0; 


[Nvox NS] = size(X);
% on full data
X = bsxfun(@minus,X,mean(X,2));
% svd -- data reduction
[U L V] = svd( X,'econ' );
Q0 = U'*X;

PRESS_set = zeros(NS,2);
kmax = NS-2;
for(iter=1:NS)

    list = randperm( NS );
    
    % split data
    Qtrn = Q0(:,list(1:end-1)); 
    Qtst = Q0(:,list(end));
    ytrn = y(list(1:end-1));
    ytst = y(list(end));
    % centering
    Qtrn   = bsxfun(@minus, Qtrn,mean(Qtrn,2)); %subtract training mean
    Qtst   = bsxfun(@minus, Qtst,mean(Qtrn,2)); %subtract training mean
    ytrn   = ytrn - mean(ytrn); %subtract training mean
    ytst   = ytst - mean(ytrn); %subtract training mean
    % re-pca
    [u2 l2 v2] = svd(Qtrn,'econ'); u2=u2(:,1:kmax);
    
    q2t = u2'*Qtrn;
    q2h = u2'*Qtst; 
    % model fit
    w0 = ( (q2t*q2t')\q2t )*ytrn;
    y_hat = q2h'*w0;
    PRESS_set(iter,1) = (y_hat - ytst)^2;
    PRESS_set(iter,2) = ytst.^2;
    PREDICT(iter) = y_hat;
end
out.PRESS = PRESS_set;
out.PREDICT = PREDICT;

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
