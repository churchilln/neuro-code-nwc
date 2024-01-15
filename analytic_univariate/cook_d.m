function [ cookScores, thresh, Pmat ] = cook_d( x, Y, ord )
%
% Cook's D: test for outliers for linear regression
%           with polynomial of order 'ord'
%
%     [ cookScores, thresh, Pmat ] = cook_d( x, Y, ord )
%
% > takes indep. vector x and a vector / column matrix of response vectors Y
% > specify polynomial order 'ord'
% 
% > returns matrix of D-values (cookScores), and recommended threshold
%   of 4/N (N=sample size / dim.1 of x and Y)
% > also returns vector / column matrix Pmat of polynomial coefficients of best
%   fit, with outliers removed

[samp respNum] = size(Y);

cookScores = zeros(samp, respNum);
Pmat       = zeros(ord+1,respNum);
thresh     = 4/(samp);

for(k=1:respNum)

	y      = Y(:,k);
	p      = polyfit(x,y,ord);
	yfit0  = polyval(p,x);
	mse    = sum( (y-yfit0).^2 )/ (samp - (ord+1));

	for(q=1:length(x))
		x2=x; x2(q)=[];
		y2=y; y2(q)=[];

		p2     = polyfit(x2,y2,ord);
		yfitQ  = polyval(p2,x);
	
		d(q) = sum( (yfit0-yfitQ).^2 )./( (length(p) )*mse );
    end

    % record cook distance values
    cookScores(:,k) = d(:);    
    % now get corrected polynomial fit
    xtmp = x;    xtmp(d>thresh) = [];
    ytmp = y;    ytmp(d>thresh) = [];
    
	ptmp = polyfit(xtmp,ytmp,ord);
    
    Pmat(:,k) = ptmp(:);
end



