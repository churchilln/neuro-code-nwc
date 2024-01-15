function LD = LD_L1_train (data, labels, Range)

% do not standardize the variables prior to analysis
standardize = false;
% dataset
T_class = sign( labels - 0.5 );
% LARS-LASSO - show the full trace of explored solutions
[Beta, A, mu, C, c, gamma] = LARS_flex(data', T_class(:), 'lars', Inf, standardize, Range+1);
% turn into col-vects of Beta weights - discard bias weights
 Beta = Beta(2:end,:)';

% % %% Currently unused: for 'lasso' flag, discard degenerate solutions
% % numVar   = ( sum(Beta~=0) );
% % transIdx = [1 numVar(2:end)-numVar(1:end-1)];
% % % keep only steps where new variable added
% % Beta = Beta(:,transIdx>0);

% keep only 1..Range steps
LD.lin_discr = Beta(:,1:Range);