function rc = RC_coef( X, Y )
%
% rv = RV_coef( X, Y )
%

X = X - repmat( mean(X), [size(X,1) 1] );
Y = Y - repmat( mean(Y), [size(Y,1) 1] );

SIG_xx = X'*X;
SIG_yy = Y'*Y;
SIG_xy = X'*Y;
SIG_yx = Y'*X;

covv = trace( SIG_xy );
vavx = trace( SIG_xx );
vavy = trace( SIG_yy );

rc = covv ./ sqrt( vavx * vavy );
