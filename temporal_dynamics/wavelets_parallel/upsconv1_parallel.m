function y = upsconv1_parallel(x,f,s,dwtARG1,dwtARG2)
%UPSCONV1 Upsample and convolution 1D.
%
%   Y = UPSCONV1(X,F_R,L,DWTATTR) returns the length-L central 
%   portion of the one step dyadic interpolation (upsample and
%    convolution) of vector X using filter F_R. The upsample 
%   and convolution attributes are described by DWTATTR.

%   M. Misiti, Y. Misiti, G. Oppenheim, J.M. Poggi 06-May-2003.
%   Last Revision: 21-May-2003.
%   Copyright 1995-2004 The MathWorks, Inc.
%   $Revision: 1.1.6.2 $  $Date: 2004/03/15 22:42:00 $

% Special case.
if isempty(x) , y = 0; return; end

% Check arguments for Extension and Shift.
switch nargin
    case 3 , 
        perFLAG  = 0;  
        dwtSHIFT = 0;
    case 4 , % Arg4 is a STRUCT
        perFLAG  = isequal(dwtARG1.extMode,'per');
        dwtSHIFT = mod(dwtARG1.shift1D,2);
    case 5 , 
        perFLAG  = isequal(dwtARG1,'per');
        dwtSHIFT = mod(dwtARG2,2);
end

% Define Length.
lx = 2*size(x,1); % vector along dim.1 (dim.2 = successive cases)
lf = length(f);   % just need one instance of filter vector
if isempty(s)
    if ~perFLAG , s = lx-lf+2; else , s = lx; end
end

% Compute Upsampling and Convolution.
y = x;
if ~perFLAG
    y = wconv1_parallel(dyadup(y,0,'r'),f);
    y = wkeep1_parallel(y,s,'c',dwtSHIFT);
else
    y = dyadup(y,0,'r',1);
    y = wextend('ar','per',y,lf/2);    
    y = wconv1_parallel(y,f);
    y = y(lf:lf+s-1,:);
    if dwtSHIFT==1 , y = y([2:end,1],:); end
end
