function [Mean, Covar] = ecmninit(Data, InitMethod)
%ECMNINIT Calculate initial mean and covariance for ECMNMLE.
%	Initial estimates for the Mean and Covariance of Data, where Data has
%	NUMSAMPLES samples of NUMSERIES random variables with missing data.
%
%	[Mean, Covar] = ecmninit(Data);
%	[Mean, Covar] = ecmninit(Data, InitMethod);
%
% Inputs:
%	Data - NUMSAMPLES x NUMSERIES matrix with NUMSAMPLES samples of a
%		NUMSERIES-dimensional random vector. Missing values are indicated
%		by NaNs. Data is the only required argument.
%
% Optional Inputs:
%	InitMethod - String to identify one of three initialization methods to
%		compute initial estimates for the mean and covariance of the data.
%		The default method is 'nanskip'. The initialization methods are:
%		'nanskip' - (default) Skip all records with NaNs.
%		'twostage' - Estimate mean, fill NaNs with mean, then estimate covar.
%		'diagonal' - Form a diagonal covar.
%
% Outputs:
%	Mean - NUMSERIES x 1 column vector initial estimate for mean of Data.
%	Covar - NUMSERIES x NUMSERIES matrix initial estimate for covariance of
%		Data.
%
% See also ECMNMLE.

%	Author(s): R.Taylor, 4-11-2005
%	Copyright 2005 The MathWorks, Inc.

% Step 1 - check arguments

if nargin < 2
	InitMethod = 'NANSKIP';
end
if nargin < 1
	error(message('finance:ecmninit:MissingInputArg'));
end
	
if isempty(Data)
    error(message('finance:ecmninit:EmptyInputData'));
end

% Step 2 - initialization

[NumSamples, NumSeries] = size(Data);

if any(sum(isnan(Data),1) == NumSamples)
	error(message('finance:ecmninit:TooManyNaNs'));
end

if sum(sum(isinf(Data)))
	error(message('finance:ecmninit:InfiniteValue'));
end

InitMethod = upper(InitMethod);
if ~any(strcmp(InitMethod,{'NANSKIP','TWOSTAGE','DIAGONAL'}))
	warning(message('finance:ecmninit:UnknownInitMethodString'));
	InitMethod = 'NANSKIP';
end

if strcmp(InitMethod,'NANSKIP') && (NumSamples - sum(any(isnan(Data),2)) <= NumSeries)
	warning(message('finance:ecmninit:ChangeInitMethod'));
	InitMethod = 'TWOSTAGE';
end

% Step 3 - estimate initial mean and covariance according to selected method

if strcmp(InitMethod,'TWOSTAGE')
	Mean = nanmean(Data)';
	Covar = zeros(NumSeries,NumSeries);
	for k = 1:NumSamples
		Z = zeros(NumSeries,1);
		for i = 1:NumSeries
			if ~isnan(Data(k,i))
				Z(i) = Data(k,i) - Mean(i);
			end
		end
		Covar = Covar + Z * Z';
	end
	Covar = (1.0/NumSamples) .* Covar;
elseif strcmp(InitMethod,'DIAGONAL')
	Mean = nanmean(Data)';
	Covar = diag(nanvar(Data,1));
else									% default method is NANSKIP
	P = ~any(isnan(Data),2);
	Mean = mean(Data(P,:))';
	Covar = cov(Data(P,:),1);
end
