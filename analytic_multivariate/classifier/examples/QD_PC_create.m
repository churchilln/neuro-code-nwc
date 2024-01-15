function QD = QD_PC_create (data, labels, numPCs)
% Quaddo
% 

data1 = data (:, labels == 1);
data0 = data (:, labels == 0);

% (1) declare
QD.mean0all = mean (data0(1:numPCs,:), 2);
QD.mean1all = mean (data1(1:numPCs,:), 2);

% (2-3) init
QD.inv_cov0all = cell(numPCs,1);
QD.inv_cov1all = cell(numPCs,1);
%
QD.logdet0all = zeros(numPCs,1);
QD.logdet1all = zeros(numPCs,1);

% (4-5) fill it up
for(k=1:numPCs)

    % calculate class covariances
    QD.inv_cov0all{k} = inv (cov (data0(1:k,:)'));
    QD.inv_cov1all{k} = inv (cov (data1(1:k,:)'));
    QD.logdet0all(k) = logdet (cov (data0(1:k,:)'));
    QD.logdet1all(k) = logdet (cov (data1(1:k,:)'));
end
