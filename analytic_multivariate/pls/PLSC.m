function out = PLSC( Xcell, Ycell, N_iters, rnk_norm, NF, fdr_q )
% .
% =========================================================================
% PLSC: script for running correlation partial least squares (PLSC),
% with permutation and bootstrap statistical testing (update: 2020/09/08)
% =========================================================================
%
% Syntax:
%          out = PLSC( Xcell, Ycell, N_iters, rnk_norm, NF, fdr_q )
%
% Input: 
%          Xcell = Gx1 cell array containing 2D matrices of  "neuroimaging" data. Each cell entry should have a data matrix of
%                  dimensions (N samples x Px) variables, corresponding to
%                  a different condition (e.g., Xcell{1}=task A, Xcell{2}=task B, ... Xcell{G}= task G )
%
%          Ycell = Gx1 cell array containing 2D matrices of  "behavioural" data being correlated with imaging data in X. Each cell entry should have a data matrix of
%                  dimensions (N samples x Py(g) ) variables, corresponding to
%                  a different condition (e.g., Ycell{1}=task A, Xcell{2}=task B, ... Xcell{G}= task G )
%
%          NB1: it is assumed that the data are repeated-measures across conditions, e.g., the nth row
%               of each data matrix ( Xcell{1}(n,:), Xcell{2}(n,:), ... Ycell{1}(n,:), Ycell{2}(n,:) ... )
%               all comes from the same subject
%          NB2: imaging data (Xcell) requires the same number of variables Px (e.g., brain regions) across all conditions.
%               however, behavioural data (Ycell) has no such constraints, and you can have different numbers of variables Py(g) for each condition
%          NB3: if no behavioural data provided (Ycell=[]) the analysis defaults to task-PLS, where Y=condition weightings 
%
%         optional args:
%
%          N_iters  = number of resampling iterations. Should be at least 1000,
%                     but more is often better - especially when looking for stable p-values
%
%          rnk_norm = choose whether to rank-normalize your data. This is rarely necessarily, but
%                     good to check robustness of your findings -- results with and without rank-normalization should roughly agree 
%                       0 = standard analysis (z-scored)
%                       1 = rank normalized (rank-transform, then z-score)
%                     Default = 0
%
%          NF       = integer, specifying number of latent factors
%                     Default = min. rank of covariance matrix
%          
%          fdq_q    = false-discovery rate threshold (0 < fdr_q < 1)
%                     Default = 0.05
%
% -------------------------------------------------------------------------
% Output:  an "out" structure with the following fields:
%
% --- 1. if you have >1 Y matrix variables
%
%    out.Xscores:    [subject x NF] matrix, each column is a vector of subject "latent variable" scores
%                                   derived from matrix X, for component f=1...NF
%    out.Yscores:    [subject x NF] matrix, each column is a vector of subject "latent variable" scores
%                                   derived from matrix Y, for component f=1...NF
%
%      out.u_bsr:         [Px x NF] matrix, each column is a vector of bootstrap ratios for loadings of variables in X. 
%                                   These values are provide z-statistic estimates of standardized effect size 
%                                   NB: they are ordered as [condition A, condition B, ... condition G]   
%      out.u_fdr:               ... matrix, applies an FDR threshold to values of u_bsr
%
%      out.v_avg: [sum(Py(g)) x NF] matrix, each column is a vector of mean loadings of variables in Y. 
%      out.v_ser:               ... matrix, each column is a vector of standard errors of loadings of variables in Y. 
%      out.v_bsr:               ... matrix, each column is a vector of bootstrap ratios for loadings of variables in Y. 
%                                   These values are provide z-statistic estimates of standardized effect size 
%      out.v_fdr:               ... matrix, applies an FDR threshold to values of v_bsr
%                                   NB: matrix columns are ordered as [condition A, condition B, ... condition G]
%
%     out.v_contr_avg [sum(Py(g)) x sum(Py(g))] matrix of mean pairwise difference between variable loadings in Y
%     out.v_contr_ser [sum(Py(g)) x sum(Py(g))] matrix of standard error on pairwise difference between variable loadings in Y
%     out.v_contr_bsr [sum(Py(g)) x sum(Py(g))] matrix of bootstrap ratios on pairwise difference between variable loadings in Y
%                                               NB: (i,j)th entry corresponds to condition(i) - condition(j)
%
%     out.sb_avg: [NFx1] vector, each entry gives variance explained by component f=1...NF
%     out.fb_avg: [NFx1] vector, each entry gives *fraction* of variance explained by component f=1...NF
%    out.sb_95ci: [NFx2] vector, each row gives 95%CI of variance explained by component f=1...NF
%    out.fb_95ci: [NFx2] vector, each row gives 95%CI of *fraction* of variance explained by component f=1...NF
%
%        out.p_l: [NFx1] vector, each entry gives permutation p-value of variance for component f=1...NF 
%       out.p_lf: [NFx1] vector, each entry gives permutation p-value of *fraction* of variance for component f=1...NF 
%

%% TASK PLS
if( isempty(Ycell) )

    % convert to cells if matrix-struct
    if(~iscell(Xcell)) Xcell = {Xcell}; end
    Ng = numel(Xcell);
    if(Ng<2) error('cannot find meaningful pls structure with <2 groups'); end
    for(g=1:Ng)
        [Nx(g,1) Px(g,1)] = size(Xcell{g});
    end
    if(numel(unique([Nx(:)]))>1)
        error('sample sizes (number of rows) must be consistent for all X matrices');
        disp('Samples in X:');
        disp(Nx');
    end
    Ns = Nx(1); clear Nx;
    if(numel(unique(Px(:)))>1)
        error('Number of variables (number of cols) must be consistent for all X matrices');
        disp('Variables in X:');
        disp(Px');
    end
    Px = Px(1);

    Pm    = min([Px, Ng-1]); % maximal rank of cross-covariance matrix
    match = double( Ng>Px ); % flag to choose which variable set to match on (0=x,1=grp)

    % checking for options / if none, use default settings
    if( nargin<3 ) N_iters  = 1000; disp('default: 1000 resampling iterations'); end
    if( nargin<4 ) rnk_norm =    2; disp('default: normalizing variance'); end
    if( nargin<5 ) NF = Pm;
                   disp(['default: model max. number of latent factors NF=',num2str(Pm)]);
    else
                   if(NF>Pm) disp('too many factors requested, adjusting...'); end
                   NF = min([NF Pm]);
                   disp(['number of latent factors NF=',num2str(NF)]);
    end
    if( nargin<6 ) fdr_q=0.05; disp('default: FDR=.05 threshold'); end

    if(match==0) ref='X'; elseif(match==1) ref='Grp'; end
    disp(['matching components based on ',ref,' data matrix']);

    % simple Imputation - replace missing values with median
    for(g=1:Ng)
    if( sum(isnan(Xcell{g}(:)))>0 )
        disp(['Found ',num2str(sum(isnan(Xcell{g}(:)))),' missing entries in X. Imputing...']);
        for(p=1:Px)
           tmp=Xcell{g}(:,p); 
           tmp(~isfinite(tmp))=median(tmp(isfinite(tmp))); 
           Xcell{g}(:,p)=tmp; 
        end
        %X = knnimpute(X')'; 
    end
    end

    % now matricize
    XMAT = [];
    GMAT = [];
    for(g=1:Ng)
        XMAT = [XMAT; Xcell{g}];
        tmp  = zeros(Ns,Ng);
        tmp(:,g)=1;
        GMAT = [GMAT; tmp];
    end
    GMAT = bsxfun(@rdivide,GMAT,sum(GMAT,1));
    idx = repmat( (1:Ns)', Ng, 1 );
    
    % if rank-normalization specified, apply to data
    if( rnk_norm == 1 )
        XMAT = tiedrank(XMAT);
    end    

    % whole-data analysis for reference 
    XMAT = bsxfun(@minus,XMAT,mean(XMAT,1));
    XMAT = bsxfun(@rdivide,XMAT,std(XMAT,0,1)+eps);
    [uo,lo,vo] = svd( XMAT'*GMAT,'econ' );
    uo=uo(:,1:NF);
    lfo=diag(lo(1:NF,1:NF))./trace(lo);
    lo=diag(lo(1:NF,1:NF));
    vo=vo(:,1:NF);
        
    %% permutation analysis

        disp('running permutation testing...');
        
        if(NF>1)

        for(iter=1:N_iters)

            disp(['resample ', num2str(iter),' of ', num2str(N_iters)]);
            GMAT_prm = GMAT;
            lsp = [0:Ns:(Ns*(Ng-1))];
            for(i=1:Ns)
                GMAT_prm( lsp+i, : ) = GMAT_prm( lsp(randperm(Ng))+i, : );
            end

            % whole-data analysis for reference 
            XMAT = bsxfun(@minus,XMAT,mean(XMAT,1));
            XMAT = bsxfun(@rdivide,XMAT,std(XMAT,0,1)+eps);
            [~,l,~] = svd( XMAT'*GMAT_prm,'econ' );
            %
            lf_prm(:,iter) = diag(l(1:NF,1:NF))./trace(l);
            l_prm(:,iter) = diag(l(1:NF,1:NF));
        end

        out.p_l  = mean( bsxfun(@gt,l_prm,lo), 2 );
        out.p_lf = mean( bsxfun(@gt,lf_prm,lfo), 2 );

        else
           
        out.p_l  = NaN;
        out.p_lf = NaN;
            
        end

    %% bootstrap analysis

    % predeclare matrices
    ub_set = zeros(Px,NF,N_iters);
    vb_set = zeros(Ng,NF,N_iters);
    sb_set = zeros(NF,N_iters);

    disp('running bootstrap testing...');
    for(iter=1:N_iters)

        disp(['resample ', num2str(iter),' of ', num2str(N_iters)]);
        list  = ceil(Ns*rand(Ns,1));
        listful = [];
        for(g=1:Ng)
           listful = [listful; list+(g-1)*Ns];
        end
        XMAT_bs = XMAT(listful,:);
        GMAT_bs = GMAT(listful,:);
        
        XMAT_bs = bsxfun(@minus,XMAT_bs,mean(XMAT_bs,1));
        XMAT_bs = bsxfun(@rdivide,XMAT_bs,std(XMAT_bs,0,1)+eps);
        [u,l,v] = svd( XMAT_bs'*GMAT_bs,'econ' );
        u=u(:,1:NF);
        lf=diag(l(1:NF,1:NF))./trace(l);
        l=diag(l(1:NF,1:NF));
        v=v(:,1:NF);

        % matching latent variable sets
        if    (match==0) %match on u (saliences of X)            
            ob=mini_procrust_ex(uo,u,'corr');
        elseif(match==1) %match on v (saliences of Y)
            ob=mini_procrust_ex(vo,v,'rss');
        end
        % store (rearranged) component sets
        ub_set(:,:,iter)=u(:,ob.index)*diag(ob.flip);
        vb_set(:,:,iter)=v(:,ob.index)*diag(ob.flip);
        sb_set(:,iter)  =l(ob.index);
        fb_set(:,iter)  =lf(ob.index);
    end

    % spatial saliences
    out.u_avg     = mean(ub_set,3);
    out.u_ser     =  std(ub_set,0,3);
    out.u_95ci     =  prctile(ub_set,[2.5 97.5],3);
    out.u_bsr  = mean(ub_set,3)./(std(ub_set,0,3)+eps);
    [p out.u_fdr] = fdr_ex( out.u_bsr, 'z', fdr_q, 0 );
    
    % task saliences
    out.v_avg     = mean(vb_set,3);
    out.v_ser     =  std(vb_set,0,3);
    out.v_95ci     =  prctile(vb_set,[2.5 97.5],3);
    out.v_bsr     = mean(vb_set,3)./std(vb_set,0,3);
    [p out.v_fdr] = fdr_ex( out.v_bsr, 'z', fdr_q, 0 );
    
    % paired contrasts
    for(f=1:NF)
        dift = bsxfun(@minus, vb_set(:,f,:), permute( vb_set(:,f,:), [2 1 3] ) );
        out.v_contr_avg(:,:,f) = mean(dift,3);
        out.v_contr_ser(:,:,f) =  std(dift,0,3);
        out.v_contr_bsr(:,:,f) = mean(dift,3)./std(dift,0,3);
    end
    
    if( NF>1 )
        
    % var fraction
    out.sb_avg    = mean(sb_set,2);
    out.sb_95ci   = prctile(sb_set,[2.5 97.5],2);
    out.fb_avg    = mean(fb_set,2);
    out.fb_95ci   = prctile(fb_set,[2.5 97.5],2);
    
    else
        
    out.sb_avg    = NaN;
    out.sb_95ci   = NaN;
    out.fb_avg    = NaN;
    out.fb_95ci   = NaN;
        
    end
    
%% CORRELATION PLS
else
    %% initialization

    % convert to cells if matrix-struct
    if(~iscell(Xcell)) Xcell = {Xcell}; end
    if(~iscell(Ycell)) Ycell = {Ycell}; end

    % data checking
    if(numel(Xcell)~=numel(Ycell))
        error('number of groups (cells) does not match for X and Y');
    end
    Ng = numel(Xcell);
    for(g=1:Ng)
        [Nx(g,1) Px(g,1)] = size(Xcell{g});
        [Ny(g,1) Py(g,1)] = size(Ycell{g});
    end
    if(numel(unique([Nx(:); Ny(:)]))>1)
        error('sample sizes (number of rows) must be consistent for all X and Y matrices');
        disp('Samples in X:');
        disp(Nx');
        disp('Samples in Y:');
        disp(Ny');
    end
    Ns = Nx(1); clear Nx Ny;
    if(numel(unique(Px(:)))>1)
        error('Number of variables (number of cols) must be consistent for all X matrices');
        disp('Variables in X:');
        disp(Px');
    end
    Px = Px(1);

    Pm    = min([Px sum(Py)]); % maximal rank of cross-covariance matrix
    match = double( sum(Py)>Px ); % flag to choose which variable set to match on (0=x,1=y)

    % checking for options / if none, use default settings
    if( nargin<3 ) N_iters  = 1000; disp('default: 1000 resampling iterations'); end
    if( nargin<4 ) rnk_norm =    0; disp('default: standard normalization'); end
    if( nargin<5 ) NF = Pm;
                   disp(['default: model max. number of latent factors NF=',num2str(Pm)]);
    else
                   if(NF>Pm) disp('too many factors requested, adjusting...'); end
                   NF = min([NF Pm]);
                   disp(['number of latent factors NF=',num2str(NF)]);
    end
    if( nargin<6 ) fdr_q=0.05; disp('default: FDR=.05 threshold'); end

    if(match==0) ref='X'; elseif(match==1) ref='Y'; end
    disp(['matching components based on ',ref,' data matrix']);

    % simple Imputation - replace missing values with median
    for(g=1:Ng)
    if( sum(isnan(Xcell{g}(:)))>0 )
        disp(['Found ',num2str(sum(isnan(Xcell{g}(:)))),' missing entries in X. Imputing...']);
        for(p=1:Px)
           tmp=Xcell{g}(:,p); 
           tmp(~isfinite(tmp))=median(tmp(isfinite(tmp))); 
           Xcell{g}(:,p)=tmp; 
        end
        %X = knnimpute(X')'; 
    end
    end
    if( sum(isnan(Ycell{g}(:)))>0 )
        disp(['Found ',num2str(sum(isnan(Ycell{g}(:)))),' missing entries in Y. Imputing...']);    
        for(p=1:Py)
           tmp=Ycell{g}(:,p); 
           tmp(~isfinite(tmp))=median(tmp(isfinite(tmp))); 
           Ycell{g}(:,p)=tmp; 
        end
        %Y = knnimpute(Y')'; 
    end

    % if rank-normalization specified, apply to data
    if( rnk_norm == 1 )
        for(g=1:Ng)
            Xcell{g} = tiedrank(Xcell{g});
            Ycell{g} = tiedrank(Ycell{g});
        end
    end    

    % whole-data analysis for reference    
    [uo,lo,lfo,vo] = svd_xprod( Xcell, Ycell, NF )

    %% permutation analysis

    if(sum(Py)>1)

        disp('running permutation testing...');

        for(iter=1:N_iters)

           disp(['resample ', num2str(iter),' of ', num2str(N_iters)]);
           listx  = randperm(Ns);
           listy  = randperm(Ns);

           for(g=1:Ng) 
               Xprm{g} = Xcell{g}(listx,:); 
               Yprm{g} = Ycell{g}(listy,:); 
           end
           [~,l_prm(:,iter),lf_prm(:,iter),~] = svd_xprod( Xprm, Yprm, NF );
        end

        out.p_l  = mean( bsxfun(@gt,l_prm,lo), 2 );
        out.p_lf = mean( bsxfun(@gt,lf_prm,lfo), 2 );
    else
        disp('cannot do permutation analysis'); 
    end

    %% bootstrap analysis

    % predeclare matrices
    ub_set = zeros(Px,NF,N_iters);
    vb_set = zeros(sum(Py),NF,N_iters);
    sb_set = zeros(NF,N_iters);

    disp('running bootstrap testing...');
    for(iter=1:N_iters)

        disp(['resample ', num2str(iter),' of ', num2str(N_iters)]);
        list  = ceil(Ns*rand(Ns,1));
        for(g=1:Ng) 
            Xbs{g} = Xcell{g}(list,:); 
            Ybs{g} = Ycell{g}(list,:); 
        end
        [u,l,lf,v] = svd_xprod( Xbs, Ybs, NF );

        % matching latent variable sets
        if    (match==0) %match on u (saliences of X)            
            ob=mini_procrust_ex(uo,u,'corr');
        elseif(match==1) %match on v (saliences of Y)
            ob=mini_procrust_ex(vo,v,'rss');
        end
        % store (rearranged) component sets
        ub_set(:,:,iter)=u(:,ob.index)*diag(ob.flip);
        vb_set(:,:,iter)=v(:,ob.index)*diag(ob.flip);
        sb_set(:,iter)  =l(ob.index);
        fb_set(:,iter)  =lf(ob.index);
    end

    % spatial saliences
    out.u_avg     = mean(ub_set,3);
    out.u_ser     =  std(ub_set,0,3);
    out.u_95ci     =  prctile(ub_set,[2.5 97.5],3);
    out.u_bsr  = mean(ub_set,3)./(std(ub_set,0,3)+eps);
    [p out.u_fdr] = fdr_ex( out.u_bsr, 'z', fdr_q, 0 );

    if(sum(Py)>1)
        % behav saliences
        out.v_avg     = mean(vb_set,3);
        out.v_ser     =  std(vb_set,0,3);
        out.v_95ci     =  prctile(vb_set,[2.5 97.5],3);
        out.v_bsr     = mean(vb_set,3)./std(vb_set,0,3);
        [p out.v_fdr] = fdr_ex( out.v_bsr, 'z', fdr_q, 0 );
        
        % paired contrasts
        for(f=1:NF)
            dift = bsxfun(@minus, vb_set(:,f,:), permute( vb_set(:,f,:), [2 1 3] ) );
            out.v_contr_avg(:,:,f) = mean(dift,3);
            out.v_contr_ser(:,:,f) =  std(dift,0,3);
            out.v_contr_bsr(:,:,f) = mean(dift,3)./std(dift,0,3);
        end
        
        % var fraction
        out.sb_avg    = mean(sb_set,2);
        out.sb_95ci   = prctile(sb_set,[2.5 97.5],2);
        out.fb_avg    = mean(fb_set,2);
        out.fb_95ci   = prctile(fb_set,[2.5 97.5],2);
    end

end

function [u,l,lf,v] = svd_xprod( Xcell, Ycell, NF )

xprod = [];
for(g=1:numel(Xcell))
    % standardized
    xtmp = bsxfun(@minus,Xcell{g},mean(Xcell{g},1));
    xtmp = bsxfun(@rdivide,xtmp,sqrt(sum(xtmp.^2,1))+eps);
    ytmp = bsxfun(@minus,Ycell{g},mean(Ycell{g},1));
    ytmp = bsxfun(@rdivide,ytmp,sqrt(sum(ytmp.^2,1))+eps);
    % concatenate along behavioural dimension
    xprod = [xprod, xtmp'*ytmp];
end

[u,l,v]=svd( xprod,'econ' );
u=u(:,1:NF);
lf=diag(l(1:NF,1:NF))./trace(l);
l=diag(l(1:NF,1:NF));
v=v(:,1:NF);

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
    CC = abs(corrcoef( [refVects, subVects] ));    
    CC = CC(1:nVct, nVct+1:end);
    
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
        %
        ordSub = [ordSub remainSub( iSub )];
        remainSub( iSub ) = [];
        %
        CCtmp(iRef,:) = [];
        CCtmp(:,iSub) = [];
    end
    
    CCnew = corrcoef( [refVects(:,ordRef) subVects(:,ordSub)] );
    flip  = sign( diag( CCnew(1:nVct,nVct+1:end) ) );
    
    resort = sortrows([ordRef(:) ordSub(:) flip(:)], 1);

    Out.index = resort(:,2);
    Out.flip   = resort(:,3);    
end

%%
function [pcritSet threshMat] = fdr_ex( dataMat, datType, qval, cv, dof )
% 
% False-Discovery Rate correction for multiple tests
% for matrix of some test statistic, get back binary matrix where
% 1=signif. at FDR thresh / 0=nonsignif
% 
%   [pcrit threshMat] = fdr( dataMat, datType, qval, cv, dof )
%
%   where datType -> 'p': p-value
%                    't': t-score -- requires a dof value
%                    'z': z-score
%
%         qval    -> level of FDR control (expected fdr rate)
% 
%         cv      -> constant of 0:[cv=1]  1:[cv= sum(1/i)]
%                    (1) applies under any joint distribution of pval
%                    (0) requires relative test indep & Gaussian noise with
%                        nonnegative correlation across tests
%         dof     -> degrees of freedom, use to assess significance of t-stat
%
%   * get back testing matrix threshMat & critical pvalue pcrit
%   * testing is currently 2-tail only!
%   * dof value is only relevant if you are using tstats
%

% ------------------------------------------------------------------------%
% Authors: Nathan Churchill, University of Toronto
%          email: nathan.churchill@rotman.baycrest.on.ca
%          Babak Afshin-Pour, Rotman reseach institute
%          email: bafshinpour@research.baycrest.org
% ------------------------------------------------------------------------%
% CODE_VERSION = '$Revision: 158 $';
% CODE_DATE    = '$Date: 2014-12-02 18:11:11 -0500 (Tue, 02 Dec 2014) $';
% ------------------------------------------------------------------------%

[Ntest Nk] = size(dataMat);

if    ( datType == 'p' )  probMat = dataMat;
elseif( datType == 't' )  probMat = 1-tcdf( abs(dataMat), dof );
elseif( datType == 'z' )  probMat = 1-normcdf( abs(dataMat) );    
end

threshMat = zeros( Ntest, Nk );

for( K=1:Nk )

    % (1) order pvalues smallest->largest
    pvect = sort( probMat(isfinite( probMat(:,K) ), K), 'ascend' );
    Ntest2= length(pvect);
    % (2) find highest index meeting limit criterion
    if(cv == 0) c_V = 1;
    else        c_V = log(Ntest2) + 0.5772 + 1/Ntest2; % approx soln.
    end
    % index vector
    indxd = (1:Ntest2)';
    % get highest index under adaptive threshold
    r = sum( pvect./indxd <= qval/(Ntest2*c_V) );

    if( r > 0 )
        % limiting p-value
        pcrit = pvect(r);
        % threshold matrix values based on prob.        
        threshMat(:,K) = double(probMat(:,K)  <= pcrit);
        % critical p-values        
        pcritSet(K,1)  = pcrit;
    else
        threshMat(:,K) = zeros(Ntest,1);
        pcritSet(K,1)  = NaN;
    end
    
end
