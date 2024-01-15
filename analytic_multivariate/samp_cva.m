function optEig = samp_cva( data1, data2, design1, design2, K_list, drf )

%% 1. initial preparation / initialization of matrices

Nvox    = size(data1,1);
% initial selection of non-transition scans
keep1   = find( sum(design1,2) > 0 );
keep2   = find( sum(design2,2) > 0 );
% combine data + ideal-file matrices
data1   = data1(:,keep1); design1 = design1(keep1,:); n_trn = size(data1,2);
data2   = data2(:,keep2); design2 = design2(keep2,:); n_tst = size(data2,2);
% concatenated data matrix + design
data    = [data1       data2];       
fulldes = [design1;  design2];

% initial SVD -> run on full dataset
[utmp s v] = svd( data'*data ); s = sqrt(s); 
full_u     = data * v * inv(s);
full_Z     = s*v'; % [pcs x time] matrix
% first round feature selection
drfPCs     = round (size (full_Z, 1) * drf);
features   = full_Z (1:drfPCs,:);
img_bases  = full_u (:,1:drfPCs);
% SVD on full data set (used for reference)
fulldata_idx1   = find (fulldes(:,1) == 1);
fulldata_idx2   = find (fulldes(:,2) == 1);
fulldata = features - repmat (mean (features, 2), [1 size(features, 2)]);
fulldata_class1 = fulldata (:, fulldata_idx1);
fulldata_class2 = fulldata (:, fulldata_idx2);
[ref_u, s, v]   = svd (fulldata, 0);
ref_Z = s*v';

% initializing data matrices...
brainLV_Zsc_sum = zeros(length (K_list), Nvox);
res_r           = zeros(length(K_list),1);
res_pp          = zeros(length(K_list),2);
res_RSPMZ       = zeros(Nvox,length(K_list));

%% 2. defining training / test data after first feature selection

% re-split data once feature selection is complete
trn_data = features(:,      1:n_trn);
tst_data = features(:,n_trn+1:end   );
% get indices for class 1/2 for fulldata & splits
fulldata_idx1  = find (fulldes(:,1) == 1);  fulldata_idx2  = find (fulldes(:,2) == 1);
new_trn_idx1   = find (design1(:,1) == 1);  new_trn_idx2   = find (design1(:,2) == 1);
new_tst_idx1   = find (design2(:,1) == 1);  new_tst_idx2   = find (design2(:,2) == 1);
% centering PCs
trn_data  = trn_data - repmat (mean (trn_data, 2), [1 n_trn]);
tst_data  = tst_data - repmat (mean (tst_data, 2), [1 n_tst]);
% partitioning by class
trn_data1 = trn_data (:, new_trn_idx1); trn_data2 = trn_data (:, new_trn_idx2);
tst_data1 = tst_data (:, new_tst_idx1); tst_data2 = tst_data (:, new_tst_idx2);
% sample sets in each split / class
n_trn_cl1 = size (trn_data1, 2);    n_trn_cl2 = size (trn_data2, 2);
n_tst_cl1 = size (tst_data1, 2);    n_tst_cl2 = size (tst_data2, 2);
% total scan number
N = n_trn + n_tst;
% svd on split matrices
[trn_u, s, v] = svd (trn_data, 0);    trn_Z = s*v';
[tst_u, s, v] = svd (tst_data, 0);    tst_Z = s*v';

%% 3. perform LD analysis for each PC-range    
    
for K_idx = 1:length(K_list)

    K = K_list (K_idx);        
%     disp( strcat( '_K=',num2str(K) ) );

    % reference set: full data (same # components!!)
    coord = ref_Z (1:K, :);
  [eigvalue, lin_discr] = LD (coord, fulldata_idx1, fulldata_idx2);
    dx_ref = ref_u (:, 1:K) * lin_discr;     % linear discriminant in image space, used as a test statistic
    ref_avg_CVscores (1) = mean (dx_ref' * fulldata_class1); % CV scores, averaged for each class
    ref_avg_CVscores (2) = mean (dx_ref' * fulldata_class2);
        
    % calculating SPM for the training set
    coord = trn_Z (1:K, :);
  [eigvalue, lin_discr] = LD (coord, new_trn_idx1, new_trn_idx2);
    dx_trn = trn_u (:, 1:K) * lin_discr;     % linear discriminant in image space, used as a test statistic
    trn_avg_CVscores (1) = mean (dx_trn' * trn_data1); % CV scores, averaged for each class
    trn_avg_CVscores (2) = mean (dx_trn' * trn_data2);
    % calculating SPM for the test set
    coord = tst_Z (1:K, :);
  [eigvalue, lin_discr] = LD (coord, new_tst_idx1, new_tst_idx2);
    dx_tst = tst_u (:, 1:K) * lin_discr;     % linear discriminant in image space, used as a test statistic
    tst_avg_CVscores (1) = mean (dx_tst' * tst_data1); % CV scores, averaged for each class
    tst_avg_CVscores (2) = mean (dx_tst' * tst_data2);

    % baby version of Procrustes transform, as implemented in NPAIRS
    sss = sum ((trn_avg_CVscores + ref_avg_CVscores).^2); % sum of squared sums
    ssd = sum ((trn_avg_CVscores - ref_avg_CVscores).^2); % sum of squared differences
    if (ssd > sss)
        dx_trn = -dx_trn;
        trn_avg_CVscores (1) = mean (dx_trn' * trn_data1);
        trn_avg_CVscores (2) = mean (dx_trn' * trn_data2);
    end
    sss = sum ((tst_avg_CVscores + ref_avg_CVscores).^2); % sum of squared sums
    ssd = sum ((tst_avg_CVscores - ref_avg_CVscores).^2); % sum of squared differences
    if (ssd > sss)
        dx_tst = -dx_tst;
        tst_avg_CVscores (1) = mean (dx_tst' * tst_data1);
        tst_avg_CVscores (2) = mean (dx_tst' * tst_data2);
    end
    
    % getting "correct" orientation on images:
    sggn = sign( ref_avg_CVscores(2)-ref_avg_CVscores(1) );

    % record correlation btw two SPMs
    map_trn = img_bases * dx_trn;
    map_tst = img_bases * dx_tst;
    r = corrcoef (map_trn, map_tst);
    spatial_corr = r (1, 2);
    
    % get mean rSPM -------------------------------------------------------

        %(1) getting the mean offsets (normed by SD)
        normedMean1 = mean(map_trn)./std(map_trn);
        normedMean2 = mean(map_tst)./std(map_tst);
        %    and rotating means into signal/noise axes
        sigMean = (normedMean1 + normedMean2)/sqrt(2);
        %noiMean = (normedMean1 - normedMean2)/sqrt(2);
        %(2) getting  signal/noise axis projections of (zscored) betamaps
        sigProj = ( zscore(map_trn) + zscore(map_tst) ) / sqrt(2);
        noiProj = ( zscore(map_trn) - zscore(map_tst) ) / sqrt(2);
        % noise-axis SD
        noiStd = std(noiProj);
        %(3) norming by noise SD:
        %     ...getting the (re-normed) mean offsets
        sigMean = sigMean./noiStd;
        %noiMean = noiMean./noiStd; 
        %     ...gettingn the normed signal/noise projection maps
        sigProj = sigProj ./ noiStd;
        %noiProj = noiProj ./ noiStd;
        % THE RSPM(Z):::
        rspmZ = sigProj + sigMean;
    
    brainLV_Zsc_sum (K_idx, :) = brainLV_Zsc_sum (K_idx, :) + rspmZ';
    % record repr + rSPM
    res_r(K_idx)       = spatial_corr;
    res_RSPMZ(:,K_idx) = res_RSPMZ(:,K_idx) + sggn * rspmZ;        

    % compute posterior probabilities
    scores_tst = dx_trn' * tst_data;
    pp_tst1_nopriors = exp (-((scores_tst - trn_avg_CVscores (1)).^2)/2);
    pp_tst2_nopriors = exp (-((scores_tst - trn_avg_CVscores (2)).^2)/2); 
   [pp_tst1_nopriors pp_tst2_nopriors] = normalize_by_sum (pp_tst1_nopriors, pp_tst2_nopriors);    
    pp_tst_trueclass_nopriors = zeros (1, n_tst);
    pp_tst_trueclass_nopriors (new_tst_idx1) = pp_tst1_nopriors (new_tst_idx1);
    pp_tst_trueclass_nopriors (new_tst_idx2) = pp_tst2_nopriors (new_tst_idx2);
    tst_pred_class_nopriors = zeros (1, n_tst) + (1);
    idx = find (pp_tst2_nopriors > pp_tst1_nopriors);
    tst_pred_class_nopriors (idx) = (2);
        
    pp_tst1_priors = pp_tst1_nopriors * (n_tst_cl1 + n_trn_cl1) / N;
    pp_tst2_priors = pp_tst2_nopriors * (n_tst_cl2 + n_trn_cl2) / N;
   [pp_tst1_priors pp_tst2_priors] = normalize_by_sum (pp_tst1_priors, pp_tst2_priors);
    pp_tst_trueclass_priors = zeros (1, n_tst);
    pp_tst_trueclass_priors (new_tst_idx1) = pp_tst1_priors (new_tst_idx1);
    pp_tst_trueclass_priors (new_tst_idx2) = pp_tst2_priors (new_tst_idx2);
    tst_pred_class_priors = zeros (1, n_tst) + (1);
    idx = find (pp_tst2_priors > pp_tst1_priors);
    tst_pred_class_priors (idx) = (2);
        
    scores_trn = dx_tst' * trn_data;
    pp_trn1_nopriors = exp (-((scores_trn - tst_avg_CVscores (1)).^2)/2);
    pp_trn2_nopriors = exp (-((scores_trn - tst_avg_CVscores (2)).^2)/2);
   [pp_trn1_nopriors pp_trn2_nopriors] = normalize_by_sum (pp_trn1_nopriors, pp_trn2_nopriors);
    pp_trn_trueclass_nopriors = zeros (1, n_trn);
    pp_trn_trueclass_nopriors (new_trn_idx1) = pp_trn1_nopriors (new_trn_idx1);
    pp_trn_trueclass_nopriors (new_trn_idx2) = pp_trn2_nopriors (new_trn_idx2);
    trn_pred_class_nopriors = zeros (1, n_trn) + (1);
    idx = find (pp_trn2_nopriors > pp_trn1_nopriors);
    trn_pred_class_nopriors (idx) = (2);

    pp_trn1_priors = pp_trn1_nopriors * (n_tst_cl1 + n_trn_cl1) / N;
    pp_trn2_priors = pp_trn2_nopriors * (n_tst_cl2 + n_trn_cl2) / N;
   [pp_trn1_priors pp_trn2_priors] = normalize_by_sum (pp_trn1_priors, pp_trn2_priors);
    pp_trn_trueclass_priors = zeros (1, n_trn);
    pp_trn_trueclass_priors (new_trn_idx1) = pp_trn1_priors (new_trn_idx1);
    pp_trn_trueclass_priors (new_trn_idx2) = pp_trn2_priors (new_trn_idx2);
    trn_pred_class_priors = zeros (1, n_trn) + (1);
    idx = find (pp_trn2_priors > pp_trn1_priors);
    trn_pred_class_priors (idx) = (2);
  
    predictMat = zeros( max([n_tst n_trn]), 2 );
    
    predictMat(1:n_tst,1) = pp_tst_trueclass_priors;
    predictMat(1:n_trn,2) = pp_trn_trueclass_priors;
    % mean, dropping off the zeros!!
    res_pp(K_idx,:)  = sum( predictMat ) ./ sum( double(predictMat > 0) );
        
end % of the K loop


% now record results for (originally output):
R     = median( res_r, 2 );
P     = median( res_pp, 2 );
D     = sqrt( (1-R).^2 + (1-P).^2 );
[v i] = min(D);

optEig  = res_RSPMZ(:,i);
