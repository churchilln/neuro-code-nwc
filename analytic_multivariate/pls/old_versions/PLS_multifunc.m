function out = PLS_multifunc( X, Y, N_iters, var_norm, NF, fdr_q )
% .
% =========================================================================
% PLS_MULTIFUNC: script for running cross-validated PLS on two
% variable sets X and Y. Uses the SVD-PLS implementation, which measures
% shared information between these matrices. Has a few different functional
% uses (hence the name)
% =========================================================================
%
% Syntax:
%          results = PLS_xvalid( X, Y, ( N_iters, var_norm, NF ) )
%
% Input: 
%          X,Y      = 2D data matrices, being compared. Should be formatted as columns matrices:
%                         X = [N samples x P1 variables]
%                         Y = [N samples x P2 variables]
%                     Note that X and Y can have different numbers of variables (columns) P1 and P2.
%                     This produces results with number of LVs:  "Nlv = min([P1 P2])"
%
%          N_iters  = number of bootstrapped resampling iterations. Should be at least 1000,
%                     but more is often better - especially when getting significance p-values
%
%          var_norm = choose whether to normalize variables to unit
%                     variance -- highly recommended if they are on different scales
%                       1 = centered (subtract variable means)
%                       2 = z-normalized (subtract mean + divide by StDev)
%                       3 = rank normalized (rank-transform, then z-normalize)
%
%          NF       = integer, specifying number of latent factors
%          
%          fdq_q    = false-discovery rate threshold (0 < fdr_q < 1)
%
% -------------------------------------------------------------------------
% Output:  an "out" structure with the following fields:
%
% --- 1. if you have >1 Y matrix variables
%
%    out.Xscores: [subject x NF] matrix, each column is a vector of subject "latent variable" scores
%                                derived from matrix X, for component f=1...NF
%    out.Yscores: [subject x NF] matrix, each column is a vector of subject "latent variable" scores
%                                derived from matrix Y, for component f=1...NF
%      out.u_bsr: [I x NF] matrix, each column is a vector of bootstrap ratios for 
%                                variable loadings, of matrix X. These are essentially
%                                Z-scores, so you can calculate Normal likelihood on each
%      out.v_bsr: [J x NF] matrix, each column is a vector of bootstrap ratios for 
%                                variable loadings, of matrix Y. Same as above
%      out.u_fdr: [I x NF] matrix, each column is a vector of binary values for 
%                                variable loadings, of matrix X. 1=significant at FDR threshold
%      out.v_fdr: [J x NF] matrix, each column is a vector of binary values for 
%                                variable loadings, of matrix Y. 1=significant at FDR threshold
%     out.sb_avg: [NFx1] vector, each entry represents variance explained by component f=1...NF
%    out.sb_frac: [NFx1] vector, each entry represents *fraction* of variance explained by component f=1...NF
%
% --- 2. if you have one Y variable
%
%         out.Xscores: [subject x 1] vector of subject "latent variable" scores
%                                     derived from matrix X
%           out.u_bsr: [I x 1] vector of bootstrap ratios for variable loadings, of 
%                                    matrix X. These are essentially Z-scores, so you can calculate Normal 
%                                    likelihood on each
%  out.Xscores_tst_avg: [subject x 1] vector of subject test scores -- unbiased version of Xscores above
%  out.Yscores_tst_avg: [subject x 1] unbiased estimates of subject Y values -- probably not useful
%  out.Xscores_tst_var: [subject x 1] vector of subject scores, reflecting resampling variability
%                                     This tells you if predicted subject score is *reliable*
%  out.Xscores_tst_err: [subject x 1] vector of subject scores, reflecting mean prediction error
%                                     This tells you if predicted subject score is *accurate*
%
% ...and if Y is continuous:
%
%             out.R2_test: mean R^2 coefficient of determination (summary test score)
%
% ...or if Y is binary:
%
%             out.PProb: mean posterior probability of correct classification
%             out.Accur: mean classification accuracy
%             out.PProb_95ci: 95% confidence interval, e.g. [2.5 to 97.5] percentiles
%             out.Accur_95ci: 95% confidence interval, e.g. [2.5 to 97.5] percentiles    

%% initialization

% checking matrix dimensions
[N1 P1] = size(X);
[N2 P2] = size(Y);
if( N1==N2 ) N=N1;
else error('number of samples (rows) must be the same in X and Y');
end

Pm = min([P1 P2 round(N*0.632)]); % smallest #variables
match = double( P2>P1 );          % flag to choose which variable set to match on (0=p1,1=p2)

% checking for options / if none, use default settings
if( nargin<3 ) N_iters  = 1000; disp('default: 1000 resampling iterations'); end
if( nargin<4 ) var_norm =    2; disp('default: normalizing variance'); end
if( nargin<5)  NF = Pm;
               disp(['default: model max. number of latent factors NF=',num2str(Pm)]);
else
               if(NF>Pm) disp('too many factors requested, adjusting...'); end
               NF = min([NF Pm]);
               disp(['number of latent factors NF=',num2str(NF)]);
end
if( size(Y,2)==1) 
    if( length(unique(Y))==2 )discrim=1; 
    else                      discrim=0;
    end
end

if(match==0) ref='X'; elseif(match==1) ref='Y'; end
disp(['matching components based on ',ref,' data matrix']);

%% Imputation - replace columns with knn
if( sum(isnan(X(:)))>0 )
    disp(['Found ',num2str(sum(isnan(X(:)))),' missing entries in X. Imputing...']);
    X = knnimpute(X')'; 
end
if( sum(isnan(Y(:)))>0 )
    disp(['Found ',num2str(sum(isnan(Y(:)))),' missing entries in Y. Imputing...']);    
    Y = knnimpute(Y')'; 
end

% if rank-normalization specified, apply to data
if( var_norm == 3 )
    X = tiedrank(X);
    Y = tiedrank(Y);
end    

% full-data reference --> for component matching
X0 = znorm( X, var_norm, 1 );
Y0 = znorm( Y, var_norm, 1 );
    
if( P2>1 )

    [u0 l0 v0] = svd( X0'*Y0, 'econ' );
     u0 = u0(:,1:NF);
     v0 = v0(:,1:NF);
     l0 = l0(1:NF,1:NF);

    out.Xscores = X0*u0;
    out.Yscores = Y0*v0;
else
    u0 = X0'*Y0;
    u0 = u0./norm(u0);
    out.Xscores = X0*u0;
end
 %% analysis 1 -- parameter reliability

% initialize parameters
%
% storing average/difference of salience parameters
ub_set = zeros( P1,NF,N_iters ); 
if(P2>1) vb_set = zeros( P2,NF,N_iters ); end
sb_set = zeros( NF,N_iters );
sb_tot = zeros(  1,N_iters );

for(i=1:N_iters) % iterate resampling splits

    disp(['resample ', num2str(i),' of ', num2str(N_iters)]);
    % randomized sampling with replacement
    list  = ceil(N*rand(N,1));
    % bootstrap samples, normalized along dim-1:
    Xb = znorm( X(list,:), var_norm, 1 );
    Yb = znorm( Y(list,:), var_norm, 1 );
    % pls decomposition via svd
    if( P2>1 )
        [u1 l1 v1] = svd( Xb'*Yb, 'econ' ); sb_tot(i) = trace(l1);
             u1 = u1(:,1:NF);
             v1 = v1(:,1:NF);
             l1 = diag(l1(1:NF,1:NF));
        % matching latent variable sets
        if    (match==0) %match on u (saliences of X)            
            %
            ob=mini_procrust_ex(u0,u1,'corr');
            %
        elseif(match==1) %match on v (saliences of Y)
            %
            ob=mini_procrust_ex(v0,v1,'rss');
            %
        end
        % store (rearranged) component sets
        ub_set(:,:,i)=u1(:,ob.index)*diag(ob.flip);
        vb_set(:,:,i)=v1(:,ob.index)*diag(ob.flip);
        sb_set(:,i)  =l1(ob.index);
    else
        u1 = Xb'*Yb;
        u1 = u1./norm(u1);
        % store (rearranged) component sets
        ub_set(:,:,i)=u1;
    end
end

out.u_bsr  = mean(ub_set,3)./std(ub_set,0,3);
[p out.u_fdr] = fdr_ex( out.u_bsr, 'z', fdr_q, 0 );

if(P2>1)
out.v_bsr     = mean(vb_set,3)./std(vb_set,0,3);
[p out.v_fdr] = fdr_ex( out.v_bsr, 'z', fdr_q, 0 );
out.sb_avg    = mean(sb_set,2);
out.sb_frac   = mean( bsxfun(@rdivide,sb_set,sb_tot),2 );
end

if(P2>1)
    for(f=1:NF)
       figure; 
       subplot(1,2,1); hold on;
       emb = mean(ub_set(:,f,:),3);
        bar( emb,'facecolor',[0.5 0.5 0.5] );
       emb = mean(ub_set(:,f,:),3); emb( abs(out.u_bsr(:,f)) < 1.64 ) = 0;
        bar( emb,'facecolor',[0.9 0.5 0.5] );
       emb = mean(ub_set(:,f,:),3); emb( abs(out.u_bsr(:,f)) < 2.32 ) = 0;
        bar( emb,'facecolor',[0.9 0.1 0.1] );
       emb = mean(ub_set(:,f,:),3); emb( abs(out.u_bsr(:,f)) < 3.09 ) = 0;
        bar( emb,'facecolor',[0.5 0.1 0.1] );
       errorbar( 1:P1, mean(ub_set(:,f,:),3), std(ub_set(:,f,:),0,3),'.k' );
       xlabel('variables');
       ylabel('loadings');   ylim( 1.5*max( abs( mean(ub_set(:,f,:),3) ) )*[-1 1] );     
       title(['F=',num2str(f),' saliences,  X matrix']);   

       subplot(1,2,2); hold on;
       emb = mean(vb_set(:,f,:),3);
        bar( emb,'facecolor',[0.5 0.5 0.5] );
       emb = mean(vb_set(:,f,:),3); emb( abs(out.v_bsr(:,f)) < 1.64 ) = 0;
        bar( emb,'facecolor',[0.9 0.5 0.5] );
       emb = mean(vb_set(:,f,:),3); emb( abs(out.v_bsr(:,f)) < 2.32 ) = 0;
        bar( emb,'facecolor',[0.9 0.1 0.1] );
       emb = mean(vb_set(:,f,:),3); emb( abs(out.v_bsr(:,f)) < 3.09 ) = 0;
        bar( emb,'facecolor',[0.5 0.1 0.1] );
    legend('non-signif.','p<.05','p<.01','p<0.001');
       errorbar( 1:P2, mean(vb_set(:,f,:),3), std(vb_set(:,f,:),0,3),'.k' );  
       xlabel('variables');
       ylabel('loadings'); ylim( 1.5*max( abs( mean(vb_set(:,f,:),3) ) )*[-1 1] );
       title(['F=',num2str(f),' saliences,  Y matrix']);   
    end
else
    
   figure; hold on;
   emb = mean(ub_set(:,1,:),3);
    bar( emb,'facecolor',[0.5 0.5 0.5] );
   emb = mean(ub_set(:,1,:),3); emb( abs(out.u_bsr(:,1)) < 1.64 ) = 0;
    bar( emb,'facecolor',[0.9 0.5 0.5] );
   emb = mean(ub_set(:,1,:),3); emb( abs(out.u_bsr(:,1)) < 2.32 ) = 0;
    bar( emb,'facecolor',[0.9 0.1 0.1] );
   emb = mean(ub_set(:,1,:),3); emb( abs(out.u_bsr(:,1)) < 3.09 ) = 0;
    bar( emb,'facecolor',[0.5 0.1 0.1] );
legend('non-signif.','p<.05','p<.01','p<0.001');    
   errorbar( 1:P1, mean(ub_set(:,1,:),3), std(ub_set(:,1,:),0,3),'.k' );
   xlabel('variables');
   ylabel('loadings');  ylim( 1.5*max( abs( mean(ub_set(:,1,:),3) ) )*[-1 1] );
       title(['F=1 saliences,  X matrix']);   
end

%% analysis 2 -- predictive power
if( size(Y,2) == 1 )
    
    disp('Single Y variable allows for prediction estimates');
    
    Ntst = max([ceil(0.1*N) 2]);
    Xscores_tst = zeros(N,N_iters);
    Yscores_tst = zeros(N,N_iters);
    
    if(discrim==0)
        
        disp('Continuous variable -- R2 coefficient estimate');

        for(i=1:N_iters) % iterate resampling splits

            disp(['resample ', num2str(i),' of ', num2str(N_iters)]);
            % randomized sampling with replacement
            list  = randperm(N);
            % test split
            Xtst = X(list(1:Ntst),:);
            Ytst = Y(list(1:Ntst),:);
            % training split
            Xtrn = X(list(Ntst+1:end),:);
            Ytrn = Y(list(Ntst+1:end),:);    
            % normalizing test data
            if( var_norm>0 )  
                Xtst = bsxfun(@minus,   Xtst,  mean(Xtrn,  1) ); 
                Ytst = bsxfun(@minus,   Ytst,  mean(Ytrn,  1) );         
            end
            if( var_norm>1 )  
                Xtst = bsxfun(@rdivide, Xtst,   std(Xtrn,0,1) ); 
                Ytst = bsxfun(@rdivide, Ytst,   std(Ytrn,0,1) );         
            end
            % normalizing training data
            Xtrn = znorm( X(list,:), var_norm, 1 );
            Ytrn = znorm( Y(list,:), var_norm, 1 );
            % pls model fit
            u1 = Xtrn'*Ytrn;
            % projection
            Xscores_tst(list(1:Ntst),i) = Xtst*u1;
            Yscores_tst(list(1:Ntst),i) = Ytst;
        end
        out.Xscores_tst_avg = sum( Xscores_tst,2 ) ./ sum( Xscores_tst~=0,2 );        
        out.Yscores_tst_avg = sum( Yscores_tst,2 ) ./ sum( Yscores_tst~=0,2 );
        %
        out.Xscores_tst_var = sum( double(Xscores_tst~=0).*bsxfun(@minus,Xscores_tst,out.Xscores_tst_avg).^2, 2 ) ./ sum( Xscores_tst~=0,2 );
        out.Xscores_tst_err = (out.Xscores_tst_avg - out.Yscores_tst_avg).^2;   
        %
        out.R2_test         = corr( out.Xscores_tst_avg, out.Yscores_tst_avg );

    elseif( discrim==1 )
        
        disp('Binary variable -- Classification accuracy estimate');        
        
        if( length(unique(Y)) ~= 2) error('discriminant vector must be binary'); end
        imn = min(Y);
        imx = max(Y);
        X1  = X(Y==imn,:); N1=sum(Y==imn); iXmn = find(Y==imn);
        X2  = X(Y==imx,:); N2=sum(Y==imx); iXmx = find(Y==imx);
        
        for(i=1:N_iters) % iterate resampling splits

            disp(['resample ', num2str(i),' of ', num2str(N_iters)]);
            % randomized sampling with replacement
            list1  = randperm(N1);
            list2  = randperm(N2);            
            % test split
            Xtst = [X1(list1(1:2),:); X2(list2(1:2),:)];
            Ytst = [imn*ones(2,1); imx*ones(2,1)];
            % training split
            Xtrn = [X1(list1(3:end),:); X2(list2(3:end),:)];
            Ytrn = [imn*ones(N1-2,1); imx*ones(N2-2,1)];
            % normalizing test data
            if( var_norm>0 )  
                Xtst = bsxfun(@minus,   Xtst,  mean(Xtrn,  1) ); 
            end
            if( var_norm>1 )  
                Xtst = bsxfun(@rdivide, Xtst,   std(Xtrn,0,1) ); 
            end
            % normalizing training data
            Xtrn = znorm( Xtrn, var_norm, 1 );
            % flipping dims.
            Xtrn = Xtrn';
            Xtst = Xtst';

            % sp1
            cl_av = [mean(Xtrn(:,Ytrn==imn),2) mean(Xtrn(:,Ytrn==imx),2)]; % class avg.
            [u l] = svd( cl_av,'econ' ); % eigen-decomp
            u_sp1 = u(:,1) * double( sign((cl_av(:,2)'*u(:,1)) - (cl_av(:,1)'*u(:,1))) ); % sign-flip
            % predict(1)
            % scores: samples x 1
            scores_clav= cl_av'* u_sp1;
            scores_sp2 = Xtst' * u_sp1;
            % pooled variance
            sig2 = ( (sum(Ytrn==imn)-1)*var(Xtrn(:,Ytrn==imn)' * u_sp1) + (sum(Ytrn==imx)-1)*var(Xtrn(:,Ytrn==imx)' * u_sp1) )./(length(Ytrn)-2); %TEMP           
            % unnormalized probabilities
            pp1_nopriors = exp(-((scores_sp2 - scores_clav(1)).^2)./(sig2*2));
            pp2_nopriors = exp(-((scores_sp2 - scores_clav(2)).^2)./(sig2*2));
            %
            pp1_priors   = pp1_nopriors .* (sum(Ytrn==imn)/length(Ytrn));
            pp2_priors   = pp2_nopriors .* (sum(Ytrn==imx)/length(Ytrn));
            % normalized
            pp1_priors_norm = pp1_priors./(pp1_priors+pp2_priors);
            pp2_priors_norm = pp2_priors./(pp1_priors+pp2_priors);
            %
            pp1_priors_norm(~isfinite(pp1_priors_norm)) = 0.50;
            pp2_priors_norm(~isfinite(pp2_priors_norm)) = 0.50;
            % probs -- sample x K-size
            PProb(i,1) = (  sum( pp1_priors_norm(Ytst==imn,:) ) + sum( pp2_priors_norm(Ytst==imx,:) )  )';
            % simple classification accuracy
            Accur(i,1) = (  sum( pp1_priors_norm(Ytst==imn,:) >0.5 ) + sum( pp2_priors_norm(Ytst==imx,:) >0.5 )  )';
            Xscores_tst(iXmn(list1(1:2)),i) = scores_sp2(1:2);    
            Xscores_tst(iXmx(list2(1:2)),i) = scores_sp2(3:4);    
        end
        out.Xscores_tst_avg = sum( Xscores_tst,2 ) ./ sum( Xscores_tst~=0,2 );
        %
        out.Xscores_tst_var = sum( double(Xscores_tst~=0).*bsxfun(@minus,Xscores_tst,out.Xscores_tst_avg).^2, 2 ) ./ sum( Xscores_tst~=0,2 );
        out.Xscores_tst_err = sum( [out.Xscores_tst_avg-mean(out.Xscores_tst_avg(Y==imn)), out.Xscores_tst_avg-mean(out.Xscores_tst_avg(Y==imx))] ...
                              .* double([Y==imn, Y==imx]), 2).^2;
        %        
        out.PProb           = sum(PProb)/(N_iters*4);
        out.Accur           = sum(Accur)/(N_iters*4);
        %
        % confidence bounds
        for(i=1:1000)
            list = ceil( length(PProb) * rand( length(PProb), 1 ) );
            pa_bsr(i,:) = [sum(PProb(list)) sum(Accur(list))]./(N_iters*4);
        end
        % 95% CIs
        out.PProb_95ci = prctile(pa_bsr(:,1),[2.5 97.5]);
        out.Accur_95ci = prctile(pa_bsr(:,2),[2.5 97.5]);        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%% figure plotting
    errat = [out.Xscores_tst_err out.Xscores_tst_var];
    errat = bsxfun(@minus,errat,min(errat));
    errat = bsxfun(@rdivide,errat,max(errat));
    figure; 
    subplot(1,3,1), imagesc( errat, [0 1]  ); colormap bone;
    ylabel('subject#');
    title('subject-specific fit');
    set(gca,'XTickLabel',{'fit-error', 'stability'})
    
    subplot(1,2,2), hold on;
    plot( Y, out.Xscores_tst_avg, 'ok', 'markerfacecolor','r' );
    xlabel('Y values'); ylabel('predicted Y values');
    if(discrim==0)
    text( min(Y)- 0.05*range(Y), max(out.Xscores_tst_avg) + 0.05*range(out.Xscores_tst_avg), ['R^2_{test}:',num2str(round(1000*out.R2_test)/1000)] );        
    else
    text( 0, max(out.Xscores_tst_avg) - 0.2*range(out.Xscores_tst_avg), ['PProb:',num2str(round(100*out.PProb)/100)] );
    text( 0, max(out.Xscores_tst_avg) - 0.5*range(out.Xscores_tst_avg), ['Accur:',num2str(round(100*out.Accur)/100)] );
    end
    xlim([min(Y)- 0.2*range(Y), max(Y)+ 0.2*range(Y)]);
    title('subject scores');
end




%% ---------------------------------------------------------- %%
function Z = znorm( X, normflag, dim )
%
% .Normalization function (can remove mean / variance, depending on requirements)
%
%  (normflag = 1) --> subtract mean
%  (normflag = 2) --> also divide out st.dev.

Z = X;

if( normflag>0 )  Z = bsxfun(@minus,   Z,  mean(Z,  dim) ); end
if( normflag>1 )  Z = bsxfun(@rdivide, Z,   std(Z,0,dim) ); end

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
