function map = QD_map_signed_set (QD, basis, data, type)
% create a sensitivity map for Fisher's quadratic discriminant
% Note: this is the new version, that does NOT take the square of the derivative
% INPUTS:
% QD -- QD structure (created by QD_create)
% basis -- PC basis (#voxels x #PCs)
% data -- data set in PC space (#PCs x #volumes)
% OUTPUT: sensitivity map (#voxels x 1)

if(strcmp(type,'NR'))

    map = zeros( size( basis,1 ), length(QD.logdet0all) );

    for(k=1:length(QD.logdet0all))

        % quadratic and linear terms of QD
        A = QD.inv_cov1all{k} - QD.inv_cov0all{k};
        b = QD.inv_cov1all{k} * QD.mean1all(1:k) - QD.inv_cov0all{k} * QD.mean0all(1:k);
        % QD is -x'Ax + 2b'x
        % Gradient of QD is -2Ax + 2b
        grad = 2* A'*data(1:k,:) + 2*repmat(b,[1 size(data,2)]);
        %
        sens_map = mean (grad, 2);
        map(:,k) = basis(:,1:k) * sens_map;
    end

elseif(strcmp(type,'L2'))

    map = zeros( size( basis,1 ), length(QD.logdet0all) );

    for(k=1:length(QD.logdet0all))

        % quadratic and linear terms of QD
        A = QD.inv_cov1all{k} - QD.inv_cov0all{k};
        b = QD.inv_cov1all{k} * QD.mean1all - QD.inv_cov0all{k} * QD.mean0all;
        % QD is -x'Ax + 2b'x
        % Gradient of QD is -2Ax + 2b
        grad = 2* A'*data + 2*repmat(b,[1 size(data,2)]);
        %
        sens_map = mean (grad, 2);
        map(:,k) = basis * sens_map;
    end    
elseif(strcmp(type,'L1'))

    map = zeros( size( basis,1 ), length(QD.logdet0all) );

    dataSub = data(QD.keptBase,:);
    
    for(k=1:length(QD.logdet0all))

        % quadratic and linear terms of QD
        A = QD.inv_cov1all{k} - QD.inv_cov0all{k};
        b = QD.inv_cov1all{k} * QD.mean1all - QD.inv_cov0all{k} * QD.mean0all;
        % QD is -x'Ax + 2b'x
        % Gradient of QD is -2Ax + 2b
        grad = 2* A'*dataSub + 2*repmat(b,[1 size(dataSub,2)]);
        %
        sens_map = mean (grad, 2);
        map(:,k) = basis(:,QD.keptBase) * sens_map;
    end    

end