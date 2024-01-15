function out = tri_pls_bootstrapped_rm( X_bar, Y, NF, deflate, rmdes )

if(nargin<5) 
     lablist = (1:size(Y,1))'; 
else lablist = rmdes;
end
lab_uniq = unique(lablist);
n_uniq   = numel(lab_uniq);

[I J K] = size(X_bar); % (subj, voxel, scale)

%allsubj -- recentering / scaling
X_bar = bsxfun(@minus,X_bar,mean(X_bar,1));
X_bar = bsxfun(@rdivide,X_bar,std(X_bar,0,1));
Y     = zscore(Y);

disp('svd running...');
xtmp = reshape( permute(X_bar,[2 1 3]), J,[],1);
[u l v] = svd(xtmp,'econ'); clear xtmp;
u    =u(:,1:end-1); P=size(u,2);
v    =v(:,1:end-1);
l    =l(1:end-1,1:end-1); 

disp('projecting down...');
qq   =l*v';
Q_bar=permute(reshape(qq,P,I,K),[2 1 3]);

disp('now doing analyses...');
[w_jq0 w_k0 T0 Q0 U0 B0 Fvar] = tri_pls12( Q_bar, Y, NF, deflate );
% out = pls_nway( Q_bar, Y, NF, [0 0], [1 1], 5, 'SVD' ); w_jq0=out.Wxv; w_k0=out.Wxm; Q0=out.Wy;
w_j0 = u*w_jq0;

for(bsr=1:500)
    bsr,
    
    %%%
    lab_uniq_boot=lab_uniq( ceil( n_uniq*rand(n_uniq,1)) ); % pick unique labels at random (w replace) 
    idx_boot =[]; % collect indices of all samples w/ allowed labels
    for(i=1:numel(lablist))
        if( sum(lablist(i)==lab_uniq_boot)>0 )
            idx_boot=[idx_boot,i];
        end
    end
    %%% 
    
    Q_bss = Q_bar(idx_boot,:,:);
    Y_bss = zscore(Y(idx_boot,:));
    [w_jq w_k T Q U B ~] = tri_pls12( Q_bss,Y_bss,NF,deflate );
%     out = pls_nway_alt( Q_bss, Y_bss, NF, [0 0], [1 1], 5, 'SVD' ); w_jq=out.Wxv; w_k=out.Wxm; Q=out.Wy;

    % matching-alt
    oo1.flip = sign(diag( (Q0'*Q) ));
    Q  = Q *diag(oo1.flip);
    w_jq = w_jq*diag(oo1.flip);  
    %
    w_j = u*w_jq;
    ooa = mini_procrust( w_j0, w_j, 'corr');
    w_jq = w_jq(:,ooa.index)*diag(ooa.flip);
    w_k = w_k(:,ooa.index)*diag(ooa.flip);
    
    w_j = u*w_jq;
    
%     % full procrustes matching...
%     w_j = u*w_jq;
%     ooa = mini_procrust( w_j0, w_j, 'corr');
%     w_jq = w_jq(:,ooa.index)*diag(ooa.flip);
%     w_k = w_k(:,ooa.index)*diag(ooa.flip);
%     Q   =   Q(:,ooa.index)*diag(ooa.flip);
%     
%     oo1.flip = sign(diag( (Q0'*Q) ));
%     Q  = Q *diag(oo1.flip);
%     w_jq = w_jq*diag(oo1.flip);    
%     
%     w_j = u*w_jq;
%     
%     %oo=mini_procrust_ex( w_j0,w_j,'corr');
%     oo.index=1:NF; oo.flip = diag( sign(corr(w_j0,w_j)));
%     %%oo=mini_procrust_ex( w_k0,w_k,'rss');
    
    w_j_set(:,:,bsr)=w_j;%(:,oo.index)*diag(oo.flip);
    w_k_set(:,:,bsr)=w_k;%(:,oo.index)*diag(oo.flip);
    Q_set(:,:,bsr)=Q;
    
end
out.w_j_bsr = mean(w_j_set,3)./std(w_j_set,0,3); %% bsr, spatial saliences
out.w_k_bsr = mean(w_k_set,3)./std(w_k_set,0,3); %% bsr, modality saliences
out.Q_bsr   = mean(Q_set  ,3)./std(Q_set  ,0,3); %% bsr, behavioural saliences
out.w_k_set = w_k_set; %% full set of modality salience values
out.Q_set   = Q_set; %% behvioural saliences
%%out.w_j_p   = 2*min( cat(3, mean( w_j_set>0, 3), mean( w_j_set<0, 3)),[],3); %% nonparametric p-values
out.w_j_av  = mean(w_j_set,3); %% mean spatial saliences
out.w_k_av  = mean(w_k_set,3); %% mean modality saliences
out.Q_av    = mean(Q_set,3); %% mean behvioural saliences

% out.Fvar = Fvar; %% covariance fraction

% %% brain - projection scores
for(i=1:I)
    for( f=1:NF) 
        out.Tscor(i,f) = out.w_j_av(:,f)'*permute( X_bar(i,:,:), [2 3 1])*out.w_k_av(:,f);
    end
end
% %% behaviour - projection scores
if( NF>1 )
    out.Uscor = Y*out.Q_av;
else
    out.Uscor=[];
end

%%
function [ Out ] = mini_procrust_ex( refVects, subVects, type )
%
% VERY simple version of procrustes - matches subVects to most appropriate
% refVect, in order to minimize global SumSquares difference criteria
%

% get dimensions from subspace-vectors
nVct     = size( subVects,2);
subV_idx = zeros(nVct,1);

if( strcmp( type , 'rss' ) )

    % get reference vector ordering, by decreasing variance
    ordRef  = sortrows( [ (1:nVct)' std( refVects )'], -2 );
    ordRef  = ordRef(:,1);

    % step through the reference vectors
    for( ir=1:nVct )
        % replicate out the reference
        tmpRef   = repmat( refVects(:,ir), [1 nVct] );
        % get the sum-of-squares difference from each reference vector (flipping to match by sign)
        SS(ir,:) = min( [sum((subVects - tmpRef).^2)', sum((-subVects - tmpRef).^2)'], [], 2 );
    end

    % we have sum-of-sqr difference matrix SS = ( nref x nsub )
    
    % step through reference vectors again (now by amt of explained var.)
    for(j=1:nVct)    
        % find the sub-vector index minimizing deviation from Ref
        [vs is] = min( SS(ordRef(j),:) );
        % for this Ref-vector, get index of best match SubVect
        subV_idx( ordRef(j) ) = is;
        % then "blank out" this option for all subsequent RefVects
        SS( :, is ) = max(SS(SS~=0)) + 1;
    end

    % reordered to match their appropriate RefVects
    subVects_reord = subVects(:,subV_idx);
    % now recontstruct what the sign was via index
    [vvv iii] = min([sum((-subVects_reord - refVects).^2)', sum((subVects_reord - refVects).^2)'], [], 2);
    % convert to actual sign
    flip= sign(iii-1.5);
    % output:
    % 
    Out.index  = subV_idx(:);
    Out.flip   = flip(:);

elseif( strcmp( type , 'corr' ) )
    
    ordRef  = (1:nVct)';

    % full correlations [ref x sub]
    CC = abs( corr( refVects, subVects ) );    
    
    remainRef = (1:nVct)'; ordRef = [];
    remainSub = (1:nVct)'; ordSub = [];
    
    CCtmp = CC;
    
    for( i=1:nVct )
        
        % get max correlation of ea. ref
        [vMax iMax] = max( CCtmp,[], 2 );
        % find Ref with highest match
        [vOpt iRef] = max( vMax     );
        % also get "sub" index:
              iSub  = iMax(iRef);
        
        ordRef = [ordRef remainRef( iRef )];
        remainRef( iRef ) = [];
        
        ordSub = [ordSub remainSub( iSub )];
        remainSub( iSub ) = [];
        
        CCtmp(iRef,:) = [];
        CCtmp(:,iSub) = [];        
    end
    
    CCnew = corr( refVects(:,ordRef), subVects(:,ordSub) );
    flip  = sign( diag( CCnew ) );
    
    resort = sortrows([ordRef(:) ordSub(:) flip(:)], 1);

    Out.index = resort(:,2);
    Out.flip   = resort(:,3);    
end
