function LR = LR_L12_create (data, labels, solvType,Range)
% create Fisher's linear discriminant, using data and labels

% initialization
T     = double( labels(:) );
X     = data';

%% "WARM START" -- estimate with a minimal pseudo-ridge regularizer %%

    % predefining model parameters
    w_pri = zeros( size(data,1), 1 )+0.01;
    %
    for( iter = 1:3 )
    
        % nb, X*w_est = [N points x 1]
        Y = sigfun( X * w_pri );
        % bernoulli probability on regressors
        rr    = Y.*(1-Y);
        % control on potentially ill-conditioned solution
        rr(rr<10e-10) = 10e-10;
        % put into matrix
        R     = diag( rr );
        % now, solving w for [Aw = z]
        XRX     = X'*R*X;
        XRz     = X'*R*X*w_pri - X'*(Y-T);
        % baseline regularizer -- 1% of smallest trace value
        LI      = 0.01* min(diag(XRX)) * eye( size(XRX,1) );
        %
        w_new = (XRX + LI)\XRz;
        % update discriminant basis
        w_pri = w_new; 
    end

%% ONWARDS!

% initialize
LR.lin_discr = zeros( size(data,1), Range );

if( strcmp( solvType, 'L2' ) )
    
    Set = exp( linspace(-10,10,Range) );
    
elseif( strcmp( solvType, 'L1' ) )
    
    Set = 1:Range;    
end
    
for(k=1:length(Set))
    
    Cval = Set(k);
    
    % predefining model parameters
    w_est = w_pri;

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
                    
        if( strcmp( solvType, 'L2' ) )
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
            
        elseif( strcmp( solvType, 'L1' ) )
            
            % formulate as OLS problem, then ridge regress   
            % basic  OLS:  ( X'*X ).^-1 * X' * t  --> regularize X'*X = X'*X + EYE*lambda
            % for logreg:  ( X'*R*X ).^-1 * X'* R * z

            % thus, change of variables Xw = sqrt(R)*X // zw = ...

            % reweighted predictors
            Xw = sqrt(R)*X;
            % reweighted predictions
            zw = sqrt(R) *X*w_est - sqrt(inv(R))*(Y-T);

            % do not standardize the variables prior to analysis
            standardize = false;
            % LARS-LASSO - show the full trace of explored solutions
            [Beta, A, mu, C, c, gamma] = LARS_flex(Xw', zw(:), 'lars', Inf, standardize, Cval);
            % turn into col-vects of Beta weights
             w_new = Beta(Cval,:)';            
        end
        
        % check for stabilized results
        % but only if >1 basis / iter > 1
        if( (k<3) || (iter==1) ) CC = 0.0;
        else      CC = corr( w_est,w_new );
        end
        
        if( (iter > 20) || (CC > 0.99) ) 
            terminFlag=1; 
        end
        % update discriminant basis
        w_est = w_new; 
    end
    
    % record optimized basis projection w
    LR.lin_discr(:,k) = w_est;
end


function s = sigfun( a )
%
% sigmoidal function:   s = 1/(1 + exp(-a) );
%
s = 1./(1 + exp(-a) );
