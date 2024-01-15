function Y = wconv1_parallel(X,f,shape)
%
% Y = wconv1_parallel(X,f,shape)
%
% convolves along columns of X
% + gives newly convolved vector set Y
%
% DONE
%

if nargin<3
    shape = 'full';
end

Y = conv2(f(:),1,X,shape);
