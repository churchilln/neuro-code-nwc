function LR = LR_PC_create (data, labels, numPCs)
% create Fisher's linear discriminant, using data and labels

% initialize
LR.lin_discr = zeros( numPCs, numPCs );

for(k=1:numPCs)
    
    % predefining model parameters
    w_est = zeros( k, 1 )+0.01;
    % initialization
    T     = double( labels(:) );
    X     = data(1:k,:)';

    iter       = 0;
    terminFlag = 0;

    while( terminFlag == 0 )
    
        iter = iter+1;

        % nb, X*w_est = [N points x 1]
        Y = sigfun( X * w_est );
        % bernoulli probability on regressors
        rr    = Y.*(1-Y);
        % control on potentially ill-conditioned solution
        rr(rr<10e-10) = 10e-10;
        % put into matrix
        R     = diag( rr );
        
        % now, solving w for [Aw = z]
        XRX     = X'*R*X;
        XRz     = X'*R*X*w_est - X'*(Y-T);
        %
        w_new = XRX\XRz;
                    
        % check for stabilized results
        % but only if >1 basis / iter > 1
        if( (k<3) || (iter==1) ) CC = 0.0;
        else      CC = corr( w_est,w_new );
        end
        
        if( (iter > 10) || (CC > 0.99) ) 
            terminFlag=1; 
        end
        % update discriminant basis
        w_est = w_new; 
    end

    % record optimized basis projection w
    LR.lin_discr(1:k,k) = w_est;
end


function s = sigfun( a )
%
% sigmoidal function:   s = 1/(1 + exp(-a) );
%
s = 1./(1 + exp(-a) );
