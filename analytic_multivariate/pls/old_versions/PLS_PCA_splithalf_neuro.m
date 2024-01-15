function out = PLS_PCA_splithalf_neuro( X, Y, N_iters, var_norm, comp_match, err_model, robust, fdr_q )
% .
% .
% =========================================================================
% PLS_PCA_splithalf_fmri:  script for running Partial Least Squares (PLS)
% of high-dimensional neuroimaging data an on adaptive Principal Component
% (PCA) basis. This model uses NPAIRS split-half cross-validation to assess
% the reliability of brain maps, and unbiased estimates of the ability to
% model brain-behaviourcorrelations.
% =========================================================================
%
% Syntax:
%              out   = PLS_PCA_splithalf_neuro( X, Y, N_iters, var_norm, comp_match, err_model, robust, fdr_q )
%
% Input: 
%         X,Y        = 2D data matrices, being compared. Should be formatted as:
%                          X = [Px variables x N samples] matrix of neuroimaging data
%                          Y = [Py variables x N samples] matrix of behavioural data
%                      Note that X and Y can have different numbers of variables (columns) Px and Py.
%                      This will produce up to Nlv = min([Px,Py]) independent PLS components. Since Py << Px
%                      for most high-dimensional neuroimaging data, usually Nlv=Py.
%
%         N_iters    = number of resampling iterations. Should be at least 100, but more is better; 
%                      you will want at least 500 for stable p-value estimates (P_pval, R_pval)
%         var_norm       = variance normalization method for imaging data
%                        0= no normalization
%                        1= voxel-space normalization
%                        2= PCA-space normalization (the default)
%                        3= both voxel and PCA-space normalization
%         comp_match = string specifying whether component matching is on voxel or behaviour saliences 
%                      uses either 'behav' or 'voxel' flags; default is 'behav'
%                      NB: this flag is only relevant for >1 behavioural measures
%         err_model  = string specifying whether noise model is global (standard NPAIRS) or voxel-specific
%                      uses either 'global' or 'voxel' flags; default is 'global'
%         robust     = binary value to determine if "robust" PLS performed. This includes outlier tests
%                      which discard outlier data at p<=0.05:
%                        0= no robust testing (default)
%                        1= perform robust testing in data space (using RV coefficient)
%                        2=perform robust testing, LV space (new!)
%                      NB1: this has not been extensively validated! Interpret with caution!! 
%                      NB2: if using this option, =2 is probably safer from corr. inflation
%         fdr_q      = False-discovery rate threshold, which is applied to PLS saliences
%                      (i.e. outputs of X_saliences and Y_saliences are set =0 if non-significant)
%                      if you are using this measure, we recommend fdr_q=0.05 (standard)
%                      If you do not want to threshold saliences, set fdr_q=0 or leave empty
%
% -------------------------------------------------------------------------
% Output:  and "out" structure with the following fields:
%
%          * this is estimated for N samples, (Px variables in X), (Py variables in Y), and Nlv latent variables
%          * we get a series of solutions estimated using different #Principal Components
%            to represent data matrix X. This allows us to optimize model flexibility to best
%            capture the underlying brain-behaviour relationship
%
%
%       out.X_saliences  = (Px x Nlv x PCS) matrix. Each column-vector "X_saliences(:,k,npc)" forms saliences
%                          of X (i.e. a brain map), for the kth component, estimated using npc Principal Components 
%       out.Y_saliences  = (Py x Nlv x PCS) matrix. Each column-vector "Y_saliences(:,k,npc)" forms saliences
%                          of Y (i.e. behaviour loadings), for the kth component, estimated using npc Principal Components 
%       out.Eigpct       = (Nlv x PCS) average percent variance explained by each component, estimated for npc Principal Components 
%
%       out.scores_x     = (N x Nlv x PCS) matrix. The column-vector scores_x(:,k,pnc) form subject
%                          latent variable "brain" scores, for the kth component, estimated using npc Principal Components 
%       out.scores_y     = (N x Nlv x PCS) matrix. The column-vector scores_y(:,k,pnc) form subject
%                          latent variable "behaviour" scores, for the kth component, estimated using npc Principal Components
%
%       out.P_behav      = (Nlv x PCS) matrix computing mean predictive correlation of each LV and Principal Component model 
%       out.R_spat       = (Nlv x PCS) matrix computing mean spatial reproducibility of each LV and Principal Component model
%       out.P_pval       = empirical significance p-value for each P_behav value
%       out.R_pval       = empirical significance p-value for each P_behav value
%
%       NOTE: we also generate output field "out.optim.(...)" which contains
%             the optimal saliences and LV scores that maximize prediction
%             and reproducibility (out.P_behav, out.R_spat)
%
%
% ------------------------------------------------------------------------%
% Author: Nathan Churchill
%  email: nchurchill.research@gmail.com
% ------------------------------------------------------------------------%
% version history: 2015/10/08
% ------------------------------------------------------------------------%
%

%% initialization - parameter checking
disp('parameter checking...');
if( nargin < 3 || isempty(N_iters)   ) disp('default N_iters=100');                N_iters    = 100;      end
if( nargin < 4 || isempty(var_norm)  ) disp('default pca-space variance norming'); var_norm   = 2;        end
if( nargin < 5 || isempty(comp_match)) disp('default matching= behav');            comp_match = 'behav';  end
if( nargin < 6 || isempty(err_model) ) disp('default err_model=global');           err_model  = 'global'; end
if( nargin < 7 || isempty(robust)    ) disp('default no outlier correction');      robust     = 0;        end
if( nargin < 8 || isempty(fdr_q)     ) disp('default no fdr');                     fdr_q      = [];       end

% checking for bad arguments
if( sum(var_norm==[0 1 2 3])==0 ) 
    error('var_norm needs to be 0,1,2,3');
end
if( ~strcmp(comp_match,'behav') && ~strcmp(comp_match,'voxel') ) 
    error('comp_match needs to be behav or voxel');
end
if( ~strcmp(err_model,'global') && ~strcmp(err_model,'voxel') ) 
    error('err_model needs to be global or voxel');
end
if    ( fdr_q >  1 ) error('fdr_q must be a positive fraction <1 (or 0=do not do)');
elseif( fdr_q <= 0 ) disp('warning: fdr<=0 sssumes no threshold!'); fdr_q=[];
end
disp('...done checks.');
disp('.');

% matrix dimensions
[Px Nx] = size(X);
[Py Ny] = size(Y);
if( Nx==Ny ) N=Nx;
else error('number of samples (rows) must be the same in X and Y');
end
% rank of split-half (size of smallest split, -1 due to mean centering)
Rsplit = floor(N/2)-1; 

% store unnormalized version of X if robust testing being done
if( robust==1 ) X_untrans = X; end

% first-stage, mean-center / normalize Y
X = bsxfun(@minus,X,mean(X,2));
if(var_norm==1 || var_norm==3)
X = bsxfun(@rdivide,X,std(X,0,2));
end
Y = bsxfun(@minus,Y,mean(Y,2));
Y = bsxfun(@rdivide,Y,std(Y,0,2));

% reduce X-matrix here
[Ufull Lfull Vfull] = svd(X,'econ');
% pc-space projection
 Qfull = Lfull*Vfull'; clear Lfull Vfull;
 
%% testing robustness - outlier influence

if( robust>0 )
   
    disp('Running robust PLS,');
    
    if    ( robust==1 )

        disp('outlier detection in data space (X)');
        disp('(delete-1 estimation of PC-space distortion, via RV coefficient)');

        [u s v] = svd( X_untrans,'econ' ); uscal_o = u*s;

        for(j=1:N)
            %
            X_tmp     =X_untrans;
            X_tmp(:,j)=[];
            [u s] = svd( X_tmp,'econ' ); uscal_del1 = u*s;
            rv(j,1) = RV_coef_ex( uscal_o, uscal_del1 );
        end

        rv_dist       = 1-rv;
        par_ab        = gamfit( rv_dist );
        p_outl        = 1-gamcdf( rv_dist, par_ab(1), par_ab(2) );
        outliers      = p_outl < 0.05; 
        outlier_index = find( outliers>0);

        figure, plot( rv_dist,'.-k' ); 
        if( sum(outliers)>0 )    
        hold on; plot( outlier_index, rv_dist(outliers>0),'or','markersize',8);
        for(j=1:length(outlier_index)) 
            text( outlier_index(j) + 0.1*outlier_index(j), rv_dist(outlier_index(j)), num2str(outlier_index(j)), 'color','r' ); 
        end
        end
        xlabel('variable X, LV#1');
        ylabel('variable Y, LV#1');
        title('Outlier points (brain data)');
        
    elseif( robust == 2 )

        disp('outlier detection in brain-behaviour space (X-Y)');
        disp('(delete-1 estimation of Mahalanobis leverage)');

        pcs = Rsplit;
        Pm  = min([Py, pcs]);

        for(n=1:N) %delete-1 testing

                % hold aside nth point, run bPLS
                Q1   = Qfull(1:pcs,:); Q1(:,n)=[];
                Qtst = Qfull(1:pcs,n) - mean(Q1,2);
                Q1   = bsxfun(@minus,Q1,mean(Q1,2));
                Y1   = Y;     Y1(:,n)=[];
                Ytst = (Y(:,n) - mean(Y1,2))./std(Y1,0,2);
                Y1 = bsxfun(@minus,Y1,mean(Y1,2));        
                Y1 = bsxfun(@rdivide,Y1,std(Y1,0,2));
                [uq1 tmp] = svd(Q1,'econ');
                QQ1 = uq1(:,1:pcs)'*Q1;
                % ============== pls ============== %
                [sal_qq1 l1 sal_y1] = svd( QQ1*Y1', 'econ' );
                 sal_q1 = uq1(:,1:pcs)*sal_qq1(:,1:Pm);
                 sal_y1 = sal_y1(:,1:Pm);
                 % training proj
                 xx_trn = (sal_q1'*Q1);
                 yy_trn = (sal_y1'*Y1);             
                 % testing proj (dim x 1)
                 xx_tst = (sal_q1'*Qtst);
                 yy_tst = (sal_y1'*Ytst);
                 ch_x(:,n) = xx_tst; %% test scores for plotting
                 ch_y(:,n) = yy_tst; %% test scores for plotting
                % Mahalanobis. per dimension
                for(w=1:Pm)
                    w,
                    xy =      [xx_tst(w)  ; yy_tst(w)  ];
                    mu = mean([xx_trn(w,:); yy_trn(w,:)],2);
                    sig= cov( [xx_trn(w,:); yy_trn(w,:)]' );
                    mahal(w,n) =  (xy-mu)'*inv(sig)*(xy-mu);
                end
        end

        KDIM       = 1; %only test for outliers along LV#1
        mahal_avg  = mean(mahal(KDIM,:),1)';
        par_ab     = gamfit( mahal_avg );
        p_outl     = 1-gamcdf( mahal_avg, par_ab(1), par_ab(2) );
        outliers   = p_outl < 0.05; 
        outlier_index = find( outliers>0);
        
        figure, plot( ch_x(1,:), ch_y(1,:),'.k' ); 
        if( sum(outliers)>0 )    
        hold on; plot( ch_x(1,outliers>0), ch_y(1,outliers>0),'or','markersize',8);
        for(j=1:length(outlier_index)) 
            text( ch_x(1,outlier_index(j)) + 0.1*ch_x(1,outlier_index(j)), ch_y(1,outlier_index(j)), num2str(outlier_index(j)), 'color','r' ); 
        end
        end
        xlabel('variable X, LV#1');
        ylabel('variable Y, LV#1');
        title('Outlier points (brain-behaviour)');
    end
%%  RECOMPUTATION WITH OUTLIERS REMOVED

    if( sum(outliers)> 0 )

        disp(['discarded ',num2str(sum(outliers)),' outlier points']);
        % trimming datapoints
        X(:,outlier_index)=[];
        Y(:,outlier_index)=[];
        N = size(X,2);
        Rsplit = floor(N/2)-1; % revised split size
        % first-stage, mean-center / normalize Y
        X = bsxfun(@minus,X,mean(X,2));
        if(var_norm==1 || var_norm==3)
        X = bsxfun(@rdivide,X,std(X,0,2));
        end        
        Y = bsxfun(@minus,Y,mean(Y,2));
        Y = bsxfun(@rdivide,Y,std(Y,0,2));
        % reduce X-matrix here
        [Ufull Lfull Vfull] = svd(X,'econ');
        % pc-space projection
        Qfull = Lfull*Vfull'; clear Lfull Vfull;
        % recording outlier points
        out.outlier_index = outlier_index;
        out.kept_index    = setdiff((1:N),outlier_index);
    else
       disp('no outliers detected!'); 
    end
end

%%
 
if(Py>1)
    % store saliences for matching later
    for(pcs=1:Rsplit)
        
        Pm  = min([Py, pcs]); % dimensionality range
        
        if    ( strcmp(comp_match,'behav') ) %store behavioural
            [sal_q0 tmp sal_y0{pcs}] = svd( Qfull(1:pcs,:)*Y', 'econ' );
        elseif( strcmp(comp_match,'voxel') ) %store voxel-maps
            [sal_q0 tmp sal_y0] = svd( Qfull(1:pcs,:)*Y', 'econ' );
            spat_x0{pcs} = Ufull(:,1:pcs) * sal_q0;
        end
    end
end

%% resampling loop

% create list of randomized splits
for(i=1:N_iters) splitlist(i,:) = randperm(N); end
% Initialization 1: prediction parameters
scores_xu = zeros(N, Py , Rsplit ); %for taking mean, at the end
scores_yv = zeros(N, Py , Rsplit ); %for taking mean, at the end
pre_corr  = zeros( Py, Rsplit, N_iters );
rep_uset  = zeros( Py, Rsplit, N_iters );
% and z-scored saliences + variance
u_zsc = zeros( Px, Py, Rsplit );
v_zsc = zeros( Py, Py, Rsplit );
s_pct = zeros( Py, Rsplit );

for(pcs=1:Rsplit) % iter. through PCs

    disp(['testing PC dimensionality ',num2str(pcs),'/',num2str(Rsplit)]);
    
    % maximum number of components
    Pm = min([Py, pcs]);    
    
    if( strcmp(err_model,'voxel'))
        % Initialization 2: average/difference of salience parameters
        u_sum = zeros( Px,Pm,1 );%just taking mean 
        u_dif = zeros( Px,Pm,N_iters ); 
        v_sum = zeros( Py,Pm,1 );%just taking mean
        v_dif = zeros( Py,Pm,N_iters );
        s_sum = zeros( Pm,1 );   %just taking mean   
        %
    elseif( strcmp(err_model,'global'))
        % Initialization 2: average/difference of salience parameters
        uz_sum = zeros( Px,Pm,1 );%one Z-scored map per loop
        v_sum  = zeros( Py,Pm,1 );%just taking mean
        v_dif  = zeros( Py,Pm,N_iters );        
        s_sum  = zeros( Pm,1 );   %just taking mean   
        %        
    end

    for(i=1:N_iters)       % iter. through resampling splits

        % Split-half selection:
        list  = splitlist(i,:); %%%MODIFIED!!
        % split X
        Q1 = Qfull(:,list(1:ceil(N/2)));
        Q2 = Qfull(:,list(ceil(N/2)+1:end));
        Q1 = bsxfun(@minus,Q1,mean(Q1,2));
        Q2 = bsxfun(@minus,Q2,mean(Q2,2));    
        % split Y
        Y1 = Y(:,list(1:ceil(N/2)));
        Y2 = Y(:,list(ceil(N/2)+1:end));
        Y1 = bsxfun(@minus,Y1,mean(Y1,2));
        Y2 = bsxfun(@minus,Y2,mean(Y2,2));
        Y1 = bsxfun(@rdivide,Y1,std(Y1,0,2));
        Y2 = bsxfun(@rdivide,Y2,std(Y2,0,2));        
        % pca decomp. of X matrices to maximize variance
        [uq1 tmp] = svd(Q1,'econ');
        [uq2 tmp] = svd(Q2,'econ');
        % projection into pc-space coordinates
        QQ1 = uq1(:,1:pcs)'*Q1;
        QQ2 = uq2(:,1:pcs)'*Q2;
        % TEST projection into pc-space coordinates
        QQ1on2 = uq2(:,1:pcs)'*Q1;
        QQ2on1 = uq1(:,1:pcs)'*Q2;
                                           
        if(var_norm==2 || var_norm==3) %%normalize on PCA-bases              
            QQ1 = bsxfun(@rdivide,QQ1,std(QQ1,0,2));
            QQ2 = bsxfun(@rdivide,QQ2,std(QQ2,0,2));  
            %
            QQ1on2 = bsxfun(@rdivide,QQ1on2,std(QQ1on2,0,2));
            QQ2on1 = bsxfun(@rdivide,QQ2on1,std(QQ2on1,0,2));          
        end
                
        if(Py>1)

            % ============== pls on split 1 ============== %
            [sal_qq1 l1 sal_y1] = svd( QQ1*Y1', 'econ' );
                 sal_x1 = Ufull*uq1(:,1:pcs)*sal_qq1(:,1:Pm);
                 sal_y1 = sal_y1(:,1:Pm);
                 l1 = diag(l1(1:Pm,1:Pm).^2); l1=l1./sum(l1); %pct var
            % ============== pls on split 2 ============== %
            [sal_qq2 l2 sal_y2] = svd( QQ2*Y2', 'econ' );
                 sal_x2 = Ufull*uq2(:,1:pcs)*sal_qq2(:,1:Pm);
                 sal_y2 = sal_y2(:,1:Pm);
                 l2 = diag(l2(1:Pm,1:Pm).^2); l2=l2./sum(l2); %pct var

            % match split-1
            if    ( strcmp(comp_match,'behav') )
                o1=mini_procrust_ex( sal_y0{pcs}(:,1:Pm),sal_y1,'rss');
            elseif( strcmp(comp_match,'voxel') )
                o1=mini_procrust_ex(spat_x0{pcs}(:,1:Pm),sal_x1,'corr');
            end
            sal_x1=sal_x1(:,o1.index)*diag(o1.flip);   
            sal_qq1=sal_qq1(:,o1.index)*diag(o1.flip);        
            sal_y1=sal_y1(:,o1.index)*diag(o1.flip);
            l1=l1(o1.index);
            % match split-2
            if    ( strcmp(comp_match,'behav') )
                o2=mini_procrust_ex( sal_y0{pcs}(:,1:Pm),sal_y2,'rss');
            elseif( strcmp(comp_match,'voxel') )
                o2=mini_procrust_ex(spat_x0{pcs}(:,1:Pm),sal_x2,'corr');
            end
            sal_x2=sal_x2(:,o2.index)*diag(o2.flip);
            sal_qq2=sal_qq2(:,o2.index)*diag(o2.flip);
            sal_y2=sal_y2(:,o2.index)*diag(o2.flip);
            l2=l2(o2.index);
        else
            % split-1
            sal_qq1 = QQ1*Y1'; sal_qq1./sqrt(sum(sal_qq1.^2));
            sal_x1  = Ufull*uq1(:,1:pcs)*sal_qq1;
            sal_y1  = 1; l1=1;
            % split-2
            sal_qq2 = QQ2*Y2'; sal_qq2./sqrt(sum(sal_qq2.^2));
            sal_x2  = Ufull*uq2(:,1:pcs)*sal_qq2;
            sal_y2  = 1; l2=1;
        end
                
        %--------------- reliability statistics
        
        if( strcmp(err_model,'voxel'))

            % store average/difference parameter estimates
            %
            u_sum = u_sum + 0.5.*(sal_x1+sal_x2);
            u_dif(:,:,i) = 0.5.*(sal_x1-sal_x2);
            %
            v_sum = v_sum + 0.5.*(sal_y1+sal_y2);
            v_dif(:,:,i) = 0.5.*(sal_y1-sal_y2);
            %
            s_sum = s_sum + 0.5.*(l1 + l2);        
            % reproducibility of paired brain saliences
            rep_uset(1:Pm,pcs,i) = diag(corr(sal_x1,sal_x2));
            
        elseif( strcmp(err_model,'global'))

            % store average/difference parameter estimates
            %
            [rep_uset(1:Pm,pcs,i), spms] = get_rSPM( sal_x1,sal_x2,1 ); %% reproducibility and maps
            uz_sum = uz_sum + spms; % add, getting overall average later
            %
            v_sum = v_sum + 0.5.*(sal_y1+sal_y2);
            v_dif(:,:,i) = 0.5.*(sal_y1-sal_y2);
            %
            s_sum = s_sum + 0.5.*(l1 + l2); 
        end
        
        %--------------- prediction statistics
        
        % unbiased LV scores: project split-1 onto split-2 model
        x1_on_u2 = QQ1on2'*sal_qq2;
        y1_on_v2 = Y1'*sal_y2;
        corr1 = diag( corr(x1_on_u2,y1_on_v2) );
        % unbiased LV scores: project split-2 onto split-1 model
        x2_on_u1 = QQ2on1'*sal_qq1;
        y2_on_v1 = Y2'*sal_y1;
        corr2 = diag( corr(x2_on_u1,y2_on_v1) );
        % average predictive correlation (how well does model predict brain-behaviour)
        pre_corr(1:Pm,pcs,i) = 0.5.*(corr1+corr2);
        
        % store actual subject scores
        tmp=zeros(N,Pm);
        %
        tmp(list(1:ceil(N/2)),    :) = x1_on_u2;
        tmp(list(ceil(N/2)+1:end),:) = x2_on_u1;
        scores_xu(:,1:Pm,pcs) = scores_xu(:,1:Pm,pcs) + tmp;
        %
        tmp(list(1:ceil(N/2)),    :) = y1_on_v2;
        tmp(list(ceil(N/2)+1:end),:) = y2_on_v1;
        scores_yv(:,1:Pm,pcs) = scores_yv(:,1:Pm,pcs) + tmp;
    end
    
    % renormalization steps
    if( strcmp(err_model,'voxel'))
        %
        % compute per-voxel zscore now
        u_zsc(:,1:Pm,pcs) = (u_sum./N_iters) ./ std(u_dif,0,3);
        %
    elseif( strcmp(err_model,'global'))
        %
        % re-scale by N_iters to get average of z-scored maps
        u_zsc(:,1:Pm,pcs) = uz_sum./N_iters;
    end
    % renormalize averages
    v_zsc(:,1:Pm,pcs) = (v_sum./N_iters) ./ std(v_dif,0,3);    
    % variance (into percentage)
    s_pct(1:Pm,pcs) = (100 .* (s_sum./N_iters));  
end
% renormalize all predicted scores
scores_xu = scores_xu./N_iters;
scores_yv = scores_yv./N_iters;

%% compiling and summarizing results

% false-discovery rate thresholding
if( ~isempty(fdr_q) )
    % step through PC bases
    for(pcs = 1:size(u_zsc,3))
        %fdr thresholding on u-dimension
        [p thr] = fdr_ex( u_zsc(:,:,pcs), 'z',fdr_q,0 ); 
        u_zsc(:,:,pcs) = u_zsc(:,:,pcs).*thr;
        %fdr thresholding on v-dimension        
        [p thr] = fdr_ex( v_zsc(:,:,pcs), 'z',fdr_q,0 ); 
        v_zsc(:,:,pcs) = v_zsc(:,:,pcs).*thr;
    end
end

% storing outputs
out.scores_x     = scores_xu;
out.scores_y     = scores_yv;
out.X_saliences  = u_zsc;
out.Y_saliences  = v_zsc;
out.Eigpct       = s_pct;
%
out.P_behav      = median(pre_corr,3);
out.R_spat       = median(rep_uset,3);
%
out.P_pval       = 1 - sum( pre_corr>0, 3)./N_iters;
out.R_pval       = 1 - sum( rep_uset>0, 3)./N_iters;

%% plotting

for(p=1:Pm) ocel{p} = ['LV#',num2str(p)]; end
figure;
subplot(1,3,1); plot( out.P_behav','.-' ); legend(ocel);
xlabel('number of PCs'); ylabel('brain-behaviour correlation');
subplot(1,3,2); plot( out.R_spat','.-' ); legend(ocel);
xlabel('number of PCs'); ylabel('reproducibility of brain pattern');
subplot(1,3,3); bar( out.Eigpct' );
xlabel('number of PCs'); ylabel('percent variance');

[vx ix] = min( (1-out.R_spat).^2 + (1-out.P_behav).^2, [],2 );
out.optim.PC_list = ix;
for(p=1:Pm)
    %
    out.optim.scores_x         = out.scores_x(:,p,ix(p));
    out.optim.scores_y         = out.scores_y(:,p,ix(p));
    %
    out.optim.X_saliences(:,p) = out.X_saliences(:,p,ix(p));
    out.optim.Y_saliences(:,p) = out.Y_saliences(:,p,ix(p));
    %
    out.optim.P_behav(p,1)     = out.P_behav(p,ix(p));
    out.optim.R_spat(p,1)      = out.R_spat(p,ix(p));
    %
    out.optim.P_pval(p,1)      = out.P_pval(p,ix(p));
    out.optim.R_pval(p,1)      = out.R_pval(p,ix(p));
    %
    out.optim.P_distro(:,p)    = permute( pre_corr(p,ix(p),:),[3 2 1]);    
    out.optim.R_distro(:,p)    = permute( rep_uset(p,ix(p),:),[3 2 1]);
end

   
%% ---------------------------------------------------------- %%
function [pcritSet threshMat] = fdr_ex( dataMat, datType, qval, cv, dof )
% 
% False-Discovery Rate correction for multiple tests

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
        pcritSet(K,1) = pcrit;
    else
        threshMat(:,K) = zeros(Ntest,1);
        pcritSet(K,1)  = NaN;
    end
    
end

%% ---------------------------------------------------------- %%
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
function [ rep, rSPM ] = get_rSPM_ex( vect1, vect2, keepMean )
%
for k =1:size(vect1,2)
    rep(k) = corr(vect1(:,k), vect2(:,k));
end

%(1) getting the mean offsets (normed by SD)
normedMean1 = mean(vect1)./std(vect1);
normedMean2 = mean(vect2)./std(vect2);
%    and rotating means into signal/noise axes
sigMean = (normedMean1 + normedMean2)/sqrt(2);
%noiMean = (normedMean1 - normedMean2)/sqrt(2);
%(2) getting  signal/noise axis projections of (zscored) betamaps
zvect1 = zscore(vect1);
zvect2 = zscore(vect2);
sigProj = ( zvect1 + zvect2)  / sqrt(2);
noiProj = ( zvect1 - zvect2)  / sqrt(2);
% noise-axis SD
noiStd = std(noiProj);
%(3) norming by noise SD:
%     ...getting the (re-normed) mean offsets
sigMean = sigMean./noiStd;
%noiMean = noiMean./noiStd; 
%  getting the normed signal/noise projection maps
sigProj = bsxfun(@rdivide,sigProj , noiStd);
%noiProj = noiProj ./ noiStd;

% Produce the rSPM:
if    ( keepMean == 1 )   rSPM = bsxfun(@plus, sigProj, sigMean);
elseif( keepMean == 0 )   rSPM = sigProj;
end

%%
function rv = RV_coef_ex( X, Y )
%
% rv = RV_coef( X, Y )
%

X = X - repmat( mean(X), [size(X,1) 1] );
Y = Y - repmat( mean(Y), [size(Y,1) 1] );

SIG_xx = X'*X;
SIG_yy = Y'*Y;
SIG_xy = X'*Y;
SIG_yx = Y'*X;

covv = trace( SIG_xy * SIG_yx );
vavx = trace( SIG_xx * SIG_xx );
vavy = trace( SIG_yy * SIG_yy );

rv = covv ./ sqrt( vavx * vavy );
