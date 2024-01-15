function LR = LR_NR_train (data, labels, Range)
% create Fisher's linear discriminant, using data and labels

% initialization
T     = double( labels(:) );
X     = data';

%% RUN LR -- estimate with a minimal pseudo-ridge regularizer %%

% predefining model parameters
w_pri = zeros( Range, 1 )+0.01;
% initialize
LR.lin_discr = zeros( Range, Range );
        
for(k=1:Range)

    % predefining model parameters
    w_est = w_pri(1:k);

    logps_est  = -99999;
    iter       = 0;
    terminFlag = 0;

    while( terminFlag == 0 )
    
        iter = iter+1;

        % nb, X*w_est = [N points x 1]
        Y = sigfun( X(:,1:k) * w_est );
        % bernoulli probability on regressors
        rr    = Y.*(1-Y);
        % control on potentially ill-conditioned solution
        rr(rr<10e-10) = 10e-10;
        % sum log prob:
        logps_new = sum( log(rr) );
        
        % put into matrix
        R     = diag( rr );
        XRX   = X(:,1:k)'*R*X(:,1:k);        
        XRz   = XRX*w_est - X(:,1:k)'*(Y-T);

        % baseline regularizer -- 1% of smallest trace value
        LI      = 0.01* min(diag(XRX)) * eye( size(XRX,1) );
        w_new   = (XRX + LI)\XRz;
                    
        % check for stabilized results
        % but only if >1 basis / iter > 1
        if  ( iter<3 ) logpdif = 1.0;
        else           logpdif = (logps_new - logps_est)./logps_est;
        end
        
        if( (iter > 30) || (logpdif < 0.05) ) 
            terminFlag=1; 
        end
        
        % update discriminant basis
        w_est = w_new; 
        % and sumlog postprob
        logps_est = logps_new;
    end
    
    % record optimized basis projection w
    LR.lin_discr(1:k,k) = w_est;
end

function s = sigfun( a )
%
% sigmoidal function:   s = 1/(1 + exp(-a) );
%
s = 1./(1 + exp(-a) );
