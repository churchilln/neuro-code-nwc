function QD = QD_L1_create (data, labels, FirstLim, Range)
% Quaddo
% 2-stage L1, to penalize on covariances

if( FirstLim > 0 )

    % do not standardize the variables prior to analysis
    standardize = false;
    % dataset
    T_class = sign( labels - 0.5 );
    % LARS-LASSO - show the full trace of explored solutions
    [Beta, A, mu, C, c, gamma] = LARS_flex(data, T_class(:), 'lars', Inf, standardize, FirstLim+1);
    % vector of 1..Firstlim components that contribute to discrim in LARS framework
     Beta = Beta(FirstLim+1,:)';
    % 
     keptBase = find (Beta~=0);
 
else
    
    keptBase = 1:size(data,1);
    
end
 
% use trimmed dataset
dataSub = data(keptBase,:);
% dataset
dataSub1 = dataSub (:, labels == 1);
dataSub0 = dataSub (:, labels == 0);

%% -- work on 1st-round feature reduced data: --

% (1) declare
QD.mean0all = mean (dataSub0, 2);
QD.mean1all = mean (dataSub1, 2);

% declare covariance, get total variance
COV_0 = cov (dataSub0'); COnrm_0 = COV_0 ./ trace(COV_0);
COV_1 = cov (dataSub1'); COnrm_1 = COV_1 ./ trace(COV_1);

LAMBDA = linspace( -15, 0, Range+1 );
LAMBDA = exp( LAMBDA(2:end) );

% (2-3) init
QD.inv_cov0all = cell(Range,1);
QD.inv_cov1all = cell(Range,1);
%
QD.logdet0all = zeros(Range,1);
QD.logdet1all = zeros(Range,1);

% (4-5) fill it up
for(k=1:Range)
        
    % -- first estimates of inverse (unscaled) --
    SINV_0 = L1precisionBCD(COnrm_0,LAMBDA(k));
    SINV_0(abs(SINV_0) < 1e-4) = 0;
    %
    SINV_1 = L1precisionBCD(COnrm_1,LAMBDA(k));
    SINV_1(abs(SINV_1) < 1e-4) = 0;
    %
    % renormulations
    SINV_0 = SINV_0./trace(COV_0);
    SINV_1 = SINV_1./trace(COV_1);
    
    % plugging into sparsemats
    
    % calculate class covariances
    QD.inv_cov0all{k} = SINV_0;
    QD.inv_cov1all{k} = SINV_1;
    QD.logdet0all(k) = -logdet (SINV_0);
    QD.logdet1all(k) = -logdet (SINV_1);
end

QD.keptBase = keptBase;