function LR = LR_L2_train (data, labels, Range)
% create Fisher's linear discriminant, using data and labels

% initialization
T     = double( labels(:) );
X     = data';

%% RUN LR -- estimate with a minimal pseudo-ridge regularizer %%

% predefining model parameters
w_pri = zeros( size(data,1), 1 )+0.01;
% initialize
LR.lin_discr = zeros( size(data,1), Range );
%
Set = exp( linspace(-10,10,Range) );
    
for(k=1:length(Set))
    
    Cval = Set(k);
    
    % predefining model parameters
    w_est = w_pri;

    logps_est  = -99999;
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
        % sum log prob:
        logps_new = sum( log(rr) );
        
        % put into matrix
        R     = diag( rr );
                    
        % formulate as OLS problem, then ridge regress   
        % basic  OLS:  ( X'*X ).^-1 * X' * t  --> regularize X'*X = X'*X + EYE*lambda
        % for logreg:  ( X'*R*X ).^-1 * X'* R * z

        % thus, change of variables Xw = sqrt(R)*X // zw = ...

        % reweighted predictors
        Xw = sqrt(R)*X;
        % reweighted predictions
        zw = sqrt(R) *X*w_est - sqrt(inv(R))*(Y-T);

        w_new = ridge(zw,Xw, Cval,0); 
        w_new = w_new(2:end);
           
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
    LR.lin_discr(:,k) = w_est;
end

function s = sigfun( a )
%
% sigmoidal function:   s = 1/(1 + exp(-a) );
%
s = 1./(1 + exp(-a) );
