function LD = LD_EN_train (data, labels, Curt, Range)

% dataset
T_class = sign( labels - 0.5 );
% LARS-LASSO - show the full trace of explored solutions
% Beta rows = [vector of predictor coefficients]
[Beta] = larsen(data', T_class(:), Curt, -Range);
 Beta=Beta( 2:end,:)'; % now each col = set of wieghts

%% Currently unused: for 'lasso' flag, discard degenerate solutions
numVar   = ( sum(Beta~=0) );
% find where new var
for(k=1:Range) 
    beta_idx(k)   = find( numVar == k, 1, 'first' );
    numVar(1:k-1) = 0; % zero out prior vars
end
% keep only steps where new variable added
Beta = Beta(:,beta_idx);
% (coefficients x steps)
LD.lin_discr = Beta;