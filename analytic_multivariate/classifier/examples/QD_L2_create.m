function QD = QD_L2_create (data, labels, subdiv)
% Quaddo
% 

data1 = data (:, labels == 1);
data0 = data (:, labels == 0);

% (1) declare
QD.mean0all = mean (data0, 2);
QD.mean1all = mean (data1, 2);

% declare covariance, get total variance
COV_0 = cov (data0');
COV_1 = cov (data1');
% define the range - total variance
[u s v] = svd( COV_0 ); Lmax1  = trace(s);
[u s v] = svd( COV_1 ); Lmax2  = trace(s);

% min scale
% Lmin  = 100*( DIG_2(1) - DIG_2(2)  ) - DIG_2(2);

% declare range of regularization
% LAMBDA = exp( linspace( log(Lmin), log(Lmax), subdiv ) );
LAMBDA = exp( linspace( -15, log(max([Lmax1 Lmax2])), subdiv ) );

% (2-3) init
QD.inv_cov0all = cell(subdiv,1);
QD.inv_cov1all = cell(subdiv,1);
%
QD.logdet0all = zeros(subdiv,1);
QD.logdet1all = zeros(subdiv,1);

% (4-5) fill it up
for(k=1:subdiv)
        
    COVreg_0 =  COV_0 + LAMBDA(k)*eye(size(data1,1)); 
    COVreg_1 =  COV_1 + LAMBDA(k)*eye(size(data1,1));         
    % calculate class covariances
    QD.inv_cov0all{k} = inv ( COVreg_0 );
    QD.inv_cov1all{k} = inv ( COVreg_1 );
    QD.logdet0all(k) = logdet (COVreg_0);
    QD.logdet1all(k) = logdet (COVreg_1);
end
