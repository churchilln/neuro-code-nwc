function results = bPLS_PET_splithalf( eigMat, behavMat, var_norm, N_iters )
% .
% =========================================================================
% BPLS_PET_SPLITHALF:  script to run "PET-style" behavioural PLS analysis 
% in split-half resampling cross-validation framework. Does estimation 
% (a) directly on brain maps (SPMs), and (b) in an optimized PCA subspace.
% Current implementation measures spatial reproducibility and predicted
% behavioural correlation; allows for multiple behavioural regressors
% =========================================================================
%
% requires (1) matrix with 1 brain SPM per subject, and
%          (2) vector with 1 behavioural measure per subject.
%
% Syntax:
%          results = bPLS_PET_splithalf( eigMat, behavMat, var_norm, N_iters, noiseEstimator )
%
% Input: 
%          eigMat   = 2D fMRI data matrix, [N voxels] x [S samples]
%          behavMat = matrix of behavioural measures, [S samples] x [B behavioural measurse]
%          var_norm = flag, to choose whether to normalize variance
%                     across samples in fmri data (eg for each voxel/PC basis) 
%                      1 = normalized (standard PLS, subtract mean + divide by StDev)
%                      0 = un-normalized (only subtract mean; may improve model performance)
%          N_iters  = # of resampling iterations
%
% -------------------------------------------------------------------------
% Output:
%       Gives results.VOX (voxel-space: estimated directly on SPMs)
%       and results.PCA (estimated in reduced PCA subspace)
%       structures with the following elements:
%
%       results.(VOX/PCA).reprod      = matrix of spatial reproducibility values, of size (N_iters x B);
%                                       each column gives values for behavioural measure b=1...B 
%       results.(VOX/PCA).corr_TEST   = matrix of predicted behav. correlation values, of size (N_iters x B)
%                                       each column gives values for behavioural measure b=1...B
%       results.(VOX/PCA).eig_GLOBAL  = column matrix of behavioural SPMs, of size(N x B) 
%       results.(VOX/PCA).score_BRAIN = column matrix of subject brain LV scores, loading on each 
%                                       of behavioural relationships b=1...B; matrix is size (S x B)
%                                       * for results.PCA, this is a cell array, where each matrix 
%                                         score_BRAIN{k} corresponds to LV scores for a PC subspace 
%                                         of size 1...k
%       results.(VOX/PCA).score_BRAIN = column matrix of subject behavioural LV scores, loading on each 
%                                       of behavioural relationships b=1...B; matrix is size (S x B)
%                                       * for results.PCA, this is a cell array, where each matrix 
%                                         score_BRAIN{k} corresponds to LV scores for a PC subspace 
%                                         of size 1...k
%                                       * this matrix is only produced for B>1 behavioural measures 
%       results.(VOX/PCA).behav_load  = a (B x B) matrix, where each column indicates behavioural variable loadings 
%                                       for the bth PLS LV.
%                                       * this matrix is only produced for B>1 behavioural measures 
% 
% ------------------------------------------------------------------------%
% Author: Nathan Churchill, University of Toronto
%  email: nathan.churchill@rotman.baycrest.on.ca
% ------------------------------------------------------------------------%
% version history: September 16 2013
% ------------------------------------------------------------------------%
%

%% (0) Preparatory Steps

% (0.0) Declaring matrix dimensions
%
% NV=#voxels, NS=#subjects, NK=max #PCs, NB=#behavioural variables
[NV NS] = size(eigMat);
 NK     = floor(NS/2);
 NB     = size(behavMat,2);
 
% (0.1) Voxel-wise bPLS initialization
%
% initialize reproducibility, correlation
COR_vox = zeros( N_iters, NB );        % reproducibility
REP_vox = zeros( N_iters, NB );        % behav. correlation
global_LV_vox = zeros( NV, NB  );      % spm LVs
LVbrainSet_vox = zeros(NS,N_iters,NB); % brain LV scores

% (0.2) PCA-space bPLS initialization
%
COR_pca = zeros( N_iters, NK, NB );     % reproducibility
REP_pca        = zeros( N_iters, NK, NB ); % behav. correlation
global_LV_pca  = zeros( NV, NK, NB);     % spm LVs
LVbrainSet_pca = cell(NK,1); for(k=1:NK) LVbrainSet_pca{k} = zeros( NS,N_iters,NB ); end % brain LV scores
        
%% ========================================================================
%% OPTION 1: single behavioural regressor
%% ========================================================================
if( NB == 1 )

    for(i=1:N_iters) % iterate resampling splits

        disp(i); 
        % Split-half selection:
        % randomize subject ordering
        list  = randperm(NS); count = round(length(list)/2);
        % separate into 2 split-halves (
        list1 = list(1:count);
        list2 = list(count+1:end);

    % ===== PREPARING DATA MATRICES ===== %

        % mean-center bootstrap and split1/2 matrices for PCA
        eigMat_1 = eigMat(:,list1);  MAT1 = znorm( eigMat_1, 1,var_norm, 2 );
        eigMat_2 = eigMat(:,list2);  MAT2 = znorm( eigMat_2, 1,var_norm, 2 );

        eigMat_1_nomean = znorm( eigMat_1, 1,0, 2 );
        eigMat_2_nomean = znorm( eigMat_2, 1,0, 2 );

        % PCA projection, split1
        [u1 s1 v1] = svd(eigMat_1_nomean'*eigMat_1_nomean ); 
         u1 = eigMat_1_nomean*v1*inv(sqrt(s1));
         Q1 = u1'*eigMat_1_nomean;
        % PCA projection, split2         
        [u2 s2 v2] = svd(eigMat_2_nomean'*eigMat_2_nomean ); 
         u2 = eigMat_2_nomean*v2*inv(sqrt(s2));
         Q2 = u2'*eigMat_2_nomean;

        Q2on1  = u1'*eigMat_2_nomean;  % project split2 onto PC-space1
        Q1on2  = u2'*eigMat_1_nomean;  % project split1 onto PC-space2

        % mean-center and norm the PCA bases
        Q1    = znorm( Q1, 1,var_norm, 2 );
        Q2    = znorm( Q2, 1,var_norm, 2 );
        %
        Q2on1 = znorm( Q2on1, 1,var_norm, 2 );
        Q1on2 = znorm( Q1on2, 1,var_norm, 2 );

        % Z-score the behaviour, separately for each split
        behvVal_1 = zscore( behavMat(list1)  );
        behvVal_2 = zscore( behavMat(list2)  );

    %% =====  VOXEL-SPACE (SPM basis) RESULTS ===== %

        % LV brain map: get projection images most collinear with behaviour:
        LV_1 = MAT1 * behvVal_1; 
        LV_2 = MAT2 * behvVal_2; 

        %% Get LV scores (train/test/bootstrap), and measure correlations %%

        % Behav Scores: unbiased (test-data) LV scores
        LV_scores_2on1 = MAT2'*LV_1;
        LV_scores_1on2 = MAT1'*LV_2;
        % Correlations put into summary matries
        COR_vox(i) = ( corr( LV_scores_2on1(:), behvVal_2(:) )  +  corr( LV_scores_1on2(:), behvVal_1(:)) )./2;

        % record the subject LV scores for plotting
        LVbrainSet_vox(list1,i) = LV_scores_1on2;
        LVbrainSet_vox(list2,i) = LV_scores_2on1;
        
        %% Estimate spatial reproducibility of LV maps %%

        % correlation between split-halves
        REP_vox(i) = corr( LV_1,LV_2 );

        % estimating reproducible SPM (rSPM) statistics
        %(1) getting the mean offsets (normed by SD)
        normedMean1 = mean(LV_1)./std(LV_1);
        normedMean2 = mean(LV_2)./std(LV_2);
        % and rotating means into signal/noise axes
        sigMean = (normedMean1 + normedMean2)/sqrt(2);
        %(2) getting  signal/noise axis projections of (zscored) betamaps
        sigProj = ( zscore(LV_1) + zscore(LV_2) ) / sqrt(2);
        noiProj = ( zscore(LV_1) - zscore(LV_2) ) / sqrt(2);
        % direct signal Z-score measurement, with global noise estimator
        global_LV_vox = global_LV_vox + ((sigProj + sigMean) ./ std(noiProj));

    %% =====  PCA-SPACE RESULTS  ===== %

        for(k=1:NK) % iterate through PC subspace sizes

            % LV brain map: get projections most collinear with behaviour, 
            % in the PCA basis space:
            projQ_1 = Q1(1:k,:)*behvVal_1;
            projQ_2 = Q2(1:k,:)*behvVal_2;
            % reconstruct LV maps in voxel spacce
            LV_1 = u1(:,1:k) * projQ_1; 
            LV_2 = u2(:,1:k) * projQ_2; 

            % Behav Scores: unbiased (test-data) LV scores
            LV_scores_2on1 = Q2on1(1:k,:)'* projQ_1;
            LV_scores_1on2 = Q1on2(1:k,:)'* projQ_2;
            % Correlations put into summary matries
            COR_pca(i,k) = ( corr( LV_scores_2on1(:), behvVal_2(:) )  +  corr( LV_scores_1on2(:), behvVal_1(:)) )./2;

            % record the subject LV scores for plotting
            LVbrainSet_pca{k}(list1,i) = LV_scores_1on2;
            LVbrainSet_pca{k}(list2,i) = LV_scores_2on1;

            %% {reproducibility estimation} %%

            % correlation between split-halves
            REP_pca(i,k) = corr( LV_1,LV_2 );

            % estimating reproducible SPM (rSPM) statistics
            %(1) getting the mean offsets (normed by SD)
            normedMean1 = mean(LV_1)./std(LV_1);
            normedMean2 = mean(LV_2)./std(LV_2);
            % and rotating means into signal/noise axes
            sigMean = (normedMean1 + normedMean2)/sqrt(2);
            %(2) getting  signal/noise axis projections of (zscored) betamaps
            sigProj = ( zscore(LV_1) + zscore(LV_2) ) / sqrt(2);
            noiProj = ( zscore(LV_1) - zscore(LV_2) ) / sqrt(2);

            % direct signal Z-score measurement, with global noise estimator
            global_LV_pca(:,k) = global_LV_pca(:,k) + ((sigProj + sigMean) ./ std(noiProj));
        end
    end

    disp('Assembling results, for 1 behavioural regressor.');

    % Taking medians on LV scores
    %
    % voxelwise
    LVbrainSet_vox = median( LVbrainSet_vox, 2 );
    % pca-wise
    for(k=1:NK) LVbrainSet_pca{k} = median( LVbrainSet_pca{k}, 2 ); end
    
    % ==========================================================
    % Results 1: voxel-space results
    % ==========================================================
    results.VOX.reprod      = REP_vox;
    results.VOX.corr_TEST   = COR_vox;
    results.VOX.score_BRAIN = LVbrainSet_vox;
    results.VOX.eig_GLOBAL  = global_LV_vox./N_iters;

    % ==========================================================
    % Results 2: pca results
    % ==========================================================
    results.PCA.reprod      = REP_pca;
    results.PCA.corr_TEST   = COR_pca;
    results.PCA.score_BRAIN = LVbrainSet_pca;
    results.PCA.eig_GLOBAL  = global_LV_pca./N_iters;

%% ========================================================================
%% OPTION 2: multiple behavioural regressors
%% ========================================================================
else
    % Extra initialization (behavioural):
    behav_load_vox= zeros( NB, NB );
    behav_load_pca= zeros( NB, NB, NK );
    %
    LVbehavSet_vox = zeros(NS,N_iters,NB);
    LVbehavSet_pca = cell(NK,1); for(k=1:NK) LVbehavSet_pca{k} = zeros( NS,N_iters,NB ); end

    % Extra step: preparing full-data (reference) representation
    %
    % z-score the behavioural values (automatic)
    behvVal_ref   = zscore( behavMat  );
    % do PCA projection using full data matrix
    % remove mean first!
    eigMat_nomean = znorm( eigMat, 1,0, 2 );
    %
    [u0 s0 v0] = svd(eigMat_nomean'*eigMat_nomean ); 
    u0 = eigMat_nomean*v0*inv(sqrt(s0));
    % full PC-space coordinates (NS bases x NS samples)
    Q_unnorm = u0'*eigMat_nomean; 
    % now, center and normalize on PC bases
    Q0 = znorm( Q_unnorm, 1, var_norm, 2 );

    CROSS   = eigMat_nomean * behvVal_ref;
    [v0 s0] = svd( CROSS'*CROSS );
    LV_ref = v0*sqrt(s0);

    for(i=1:N_iters) % iterate resampling splits

        disp(i); 
        % Split-half selection:
        % randomize subject ordering
        list  = randperm(NS); count = round(length(list)/2);
        % separate into 2 split-halves (
        list1 = list(1:count);
        list2 = list(count+1:end);

    % ===== PREPARING DATA MATRICES ===== %

        % mean-center bootstrap and split1/2 matrices for PCA
        eigMat_1 = eigMat(:,list1);  MAT1 = znorm( eigMat_1, 1,var_norm, 2 );
        eigMat_2 = eigMat(:,list2);  MAT2 = znorm( eigMat_2, 1,var_norm, 2 );

        eigMat_1_nomean = znorm( eigMat_1, 1,0, 2 );
        eigMat_2_nomean = znorm( eigMat_2, 1,0, 2 );

        % PCA projection, split1
        [u1 s1 v1] = svd(eigMat_1_nomean'*eigMat_1_nomean ); 
         u1 = eigMat_1_nomean*v1*inv(sqrt(s1));
         Q1 = u1'*eigMat_1_nomean;
        % PCA projection, split2         
        [u2 s2 v2] = svd(eigMat_2_nomean'*eigMat_2_nomean ); 
         u2 = eigMat_2_nomean*v2*inv(sqrt(s2));
         Q2 = u2'*eigMat_2_nomean;

        Q2on1  = u1'*eigMat_2_nomean;  % project split2 onto PC-space1
        Q1on2  = u2'*eigMat_1_nomean;  % project split1 onto PC-space2

        % mean-center and norm the PCA bases
        Q1    = znorm( Q1, 1,var_norm, 2 );
        Q2    = znorm( Q2, 1,var_norm, 2 );
        %
        Q2on1 = znorm( Q2on1, 1,var_norm, 2 );
        Q1on2 = znorm( Q1on2, 1,var_norm, 2 );

        % Z-score the behaviour, separately for each split
        behvVal_1 = zscore( behavMat(list1,:)  );
        behvVal_2 = zscore( behavMat(list2,:)  );

    %% =====  VOXEL-SPACE (SPM basis) RESULTS ===== %

        % LV brain map: get projection images most collinear with behaviour:
        CROSS1 = MAT1 * behvVal_1; 
        CROSS2 = MAT2 * behvVal_2; 
        %
        [vx1 sx1] = svd( CROSS1'*CROSS1 );  ux1=CROSS1*vx1*inv(sqrt(sx1));
        [vx2 sx2] = svd( CROSS2'*CROSS2 );  ux2=CROSS2*vx2*inv(sqrt(sx2));

        [ Out1 ] = mini_procrust_ex( LV_ref, vx1*sqrt(sx1), 'rss' );
        [ Out2 ] = mini_procrust_ex( LV_ref, vx2*sqrt(sx2), 'rss' );

        vx1 = vx1(:,Out1.index) * diag( Out1.flip );
        ux1 = ux1(:,Out1.index) * diag( Out1.flip );
        %
        vx2 = vx2(:,Out2.index) * diag( Out2.flip );
        ux2 = ux2(:,Out2.index) * diag( Out2.flip );

         behav_load_vox = behav_load_vox + (vx1+vx2)./2;

        %% Get LV scores (train/test/bootstrap), and measure correlations %%

        % Behav Scores: unbiased (test-data) LV scores
        scor_brain = MAT2'*ux1;
        scor_behav = behvVal_2*vx1;
        cor2on1    = diag( corr( scor_brain, scor_behav ) );
        %---
        LVbrainSet_vox(list2,i,:) = scor_brain;
        LVbehavSet_vox(list2,i,:) = scor_behav;

        scor_brain = MAT1'*ux2;
        scor_behav = behvVal_1*vx2;
        cor1on2    = diag( corr( scor_brain, scor_behav ) );
        %---
        LVbrainSet_vox(list1,i,:) = scor_brain;
        LVbehavSet_vox(list1,i,:) = scor_behav;

        COR_vox(i,:) = ( cor2on1 + cor1on2 )./2;

        %% Estimate spatial reproducibility of LV maps %%
        LV_spat_1 = ux1;
        LV_spat_2 = ux2;
        % correlation between split-halves
        REP_vox(i,:) = diag(corr( LV_spat_1,LV_spat_2 ));
        % estimating reproducible SPM (rSPM) statistics
        %(1) getting the mean offsets (normed by SD)
        normedMean1 = mean(LV_spat_1)./std(LV_spat_1);
        normedMean2 = mean(LV_spat_2)./std(LV_spat_2);
        % and rotating means into signal/noise axes
        sigMean = (normedMean1 + normedMean2)/sqrt(2);
        %(2) getting  signal/noise axis projections of (zscored) betamaps
        sigProj = ( zscore(LV_spat_1) + zscore(LV_spat_2) ) ./ sqrt(2);
        noiProj = ( zscore(LV_spat_1) - zscore(LV_spat_2) ) ./ sqrt(2);
        % direct signal Z-score measurement, with global noise estimator
        global_LV_vox = global_LV_vox + (sigProj + repmat( sigMean, [NV 1] ) ) ./ repmat( std(noiProj), [NV 1] );

    %% =====  PCA-SPACE RESULTS  ===== %

        for(k=NB:NK) % iterate through PC subspace sizes

            % LV brain map: get projections most collinear with behaviour, 
            % in the PCA basis space:
            CROSS1 = Q1(1:k,:)*behvVal_1;
            CROSS2 = Q2(1:k,:)*behvVal_2;
            %
            [ux1 sx1 vx1] = svd( CROSS1,'econ' );
            [ux2 sx2 vx2] = svd( CROSS2,'econ' );

            [ Out1 ] = mini_procrust_ex( LV_ref, vx1*sx1, 'rss' );
            [ Out2 ] = mini_procrust_ex( LV_ref, vx2*sx2, 'rss' );

            vx1 = vx1(:,Out1.index) * diag( Out1.flip );
            ux1 = ux1(:,Out1.index) * diag( Out1.flip );
            %
            vx2 = vx2(:,Out2.index) * diag( Out2.flip );
            ux2 = ux2(:,Out2.index) * diag( Out2.flip );

            behav_load_pca(:,:,k) = behav_load_pca(:,:,k) + (vx1+vx2)./2;

            % Behav Scores: unbiased (test-data) LV scores
            scor_brain = Q2on1(1:k,:)'*ux1;
            scor_behav = behvVal_2*vx1;
            cor2on1    = diag( corr( scor_brain, scor_behav ) );
            %---
            LVbrainSet_pca{k}(list2,:,i) = scor_brain;
            LVbehavSet_pca{k}(list2,:,i) = scor_behav;

            scor_brain = Q1on2(1:k,:)'*ux2;
            scor_behav = behvVal_1*vx2;
            cor1on2    = diag( corr( scor_brain, scor_behav ) );
            %---
            LVbrainSet_pca{k}(list1,i,:) = scor_brain;
            LVbehavSet_pca{k}(list1,i,:) = scor_behav;

            COR_pca(i,k,:) = ( cor2on1 + cor1on2 )./2;

            %% Estimate spatial reproducibility of LV maps %%
            LV_spat_1 = u1(:,1:k) * ux1;
            LV_spat_2 = u2(:,1:k) * ux2;

            % correlation between split-halves
            REP_pca(i,k,:) = diag(corr( LV_spat_1,LV_spat_2 ));

            % estimating reproducible SPM (rSPM) statistics
            %(1) getting the mean offsets (normed by SD)
            normedMean1 = mean(LV_spat_1)./std(LV_spat_1);
            normedMean2 = mean(LV_spat_2)./std(LV_spat_2);
            % and rotating means into signal/noise axes
            sigMean = (normedMean1 + normedMean2)/sqrt(2);
            %(2) getting  signal/noise axis projections of (zscored) betamaps
            sigProj = ( zscore(LV_spat_1) + zscore(LV_spat_2) ) ./ sqrt(2);
            noiProj = ( zscore(LV_spat_1) - zscore(LV_spat_2) ) ./ sqrt(2);

            for(b=1:NB)
                spm = (sigProj(:,b) + sigMean(b))./std(noiProj(:,b));                
                % direct signal Z-score measurement, with global noise estimator
                global_LV_pca(:,k,b) = global_LV_pca(:,k,b) + spm;
            end
        end
    end

    disp('Assembling results, for multiple behavioural regressors.');

    % Taking medians on LV scores
    %
    % voxelwise
    LVbrainSet_vox = permute( median( LVbrainSet_vox, 2 ), [1 3 2] );
    LVbehavSet_vox = permute( median( LVbehavSet_vox, 2 ), [1 3 2] );
    % pca-wise
    for(k=1:NK) LVbrainSet_pca{k} = permute( median( LVbrainSet_pca{k}, 2 ), [1 3 2] ); end
    for(k=1:NK) LVbehavSet_pca{k} = permute( median( LVbehavSet_pca{k}, 2 ), [1 3 2] ); end
    
    % ==========================================================
    % Results 1: voxel-space results
    % ==========================================================
    results.VOX.reprod      = REP_vox;
    results.VOX.corr_TEST   = COR_vox;
    results.VOX.score_BRAIN = LVbrainSet_vox;
    results.VOX.score_BEHAV = LVbehavSet_vox;
    results.VOX.eig_GLOBAL  = global_LV_vox./N_iters;
    results.VOX.behav_load  = behav_load_vox;

    % ==========================================================
    % Results 2: pca results
    % ==========================================================
    results.PCA.reprod      = REP_pca;
    results.PCA.corr_TEST   = COR_pca;
    results.PCA.score_BRAIN = LVbrainSet_pca;
    results.PCA.score_BEHAV = LVbehavSet_pca;
    results.PCA.eig_GLOBAL  = global_LV_pca./N_iters;
    results.PCA.behav_load  = permute( behav_load_pca, [1 3 2] );
end

%% ---------------------------------------------------------- %%
function Z = znorm( X, avgFlag, varFlag, dim )
%
% .Normalization function (can remove mean / variance, depending on requirements)
%

Z = X;
alldims = size(Z);          % #elements at each dimension
numdims = length(alldims);  % number of dimensions
repdim  = ones(numdims,1);  % building #dims for replication
repdim(dim) = alldims(dim); % ...

if( avgFlag>0 )  Z = Z  - repmat(  mean(Z,dim), repdim ); end
if( varFlag>0 )  Z = Z ./ repmat( std(Z,0,dim), repdim ); end

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