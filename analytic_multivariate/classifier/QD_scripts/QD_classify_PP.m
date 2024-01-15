function [accuracy] = QD_classify_PP (QD, basis, train_means, test_data, test_labels, type);

% differences = = baswt x scan
DIF_0 = basis' * ( test_data - repmat( train_means.mean0, [1 size(test_data,2)] ) );
DIF_1 = basis' * ( test_data - repmat( train_means.mean1, [1 size(test_data,2)] ) );

if(strcmp(type,'NR'))

    accuracy = zeros( length(QD.mean1all), 1 );

    for(k=1:length(QD.mean1all))
        %
        pp0  = exp( -0.5 * diag( DIF_0(1:k,:)'*QD.inv_cov0all{k}*DIF_0(1:k,:) ) - 0.5*QD.logdet0all(k) );
        pp1  = exp( -0.5 * diag( DIF_1(1:k,:)'*QD.inv_cov1all{k}*DIF_1(1:k,:) ) - 0.5*QD.logdet1all(k) );
        %
        pp0n = pp0./(pp0+pp1);
        pp1n = pp1./(pp0+pp1);
        
        pp0n( ~isfinite(pp0n) ) = 0.5;
        pp1n( ~isfinite(pp1n) ) = 0.5;
        
        %
        accuracy(k) = mean( [pp0n(test_labels(:)==0); pp1n(test_labels(:)==1)] );
    end


elseif(strcmp(type,'L2'))
    
    accuracy = zeros( length(QD.logdet0all), 1 );

    for(k=1:length(QD.logdet0all))
        %
        pp0 = exp( -0.5 * diag( DIF_0'*QD.inv_cov0all{k}*DIF_0 ) - 0.5*QD.logdet0all(k) );
        pp1 = exp( -0.5 * diag( DIF_1'*QD.inv_cov1all{k}*DIF_1 ) - 0.5*QD.logdet1all(k) );
        %
        pp0n = pp0./(pp0+pp1);
        pp1n = pp1./(pp0+pp1);
        
        pp0n( ~isfinite(pp0n) ) = 0.5;
        pp1n( ~isfinite(pp1n) ) = 0.5;

        %
        accuracy(k) = mean( [pp0n(test_labels(:)==0); pp1n(test_labels(:)==1)] );
    end

 elseif(strcmp(type,'L1'))

     DIF_0 = DIF_0(QD.keptBase,:);
     DIF_1 = DIF_1(QD.keptBase,:);
     
    accuracy = zeros( length(QD.logdet0all), 1 );

    for(k=1:length(QD.logdet0all))
        %
        pp0 = exp( -0.5 * diag( DIF_0'*QD.inv_cov0all{k}*DIF_0 ) - 0.5*QD.logdet0all(k) );
        pp1 = exp( -0.5 * diag( DIF_1'*QD.inv_cov1all{k}*DIF_1 ) - 0.5*QD.logdet1all(k) );
        %
        pp0n = pp0./(pp0+pp1);
        pp1n = pp1./(pp0+pp1);
        
        pp0n( ~isfinite(pp0n) ) = 0.5;
        pp1n( ~isfinite(pp1n) ) = 0.5;
        
        %
        accuracy(k) = mean( [pp0n(test_labels(:)==0); pp1n(test_labels(:)==1)] );        
    end
end
