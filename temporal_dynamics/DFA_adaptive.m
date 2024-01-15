function [out] = DFA_adaptive( DataMat, varargin )
%
% -------------------------------------------------------------------------
% DETRENDED FLUCTUATIONS ANALYSIS: very simple, (relatively) computationally
% efficient code for computing the Hurst exponent simultaneously for a
% matrix of time-series vectors. This measures the long-range temporal
% dependency of your signal estimates, and is robust against linear signal
% drift effects.
% -------------------------------------------------------------------------
%
% Syntax:
%            [H_vect log_FVAR scale_list] = DFA_parallel( DataMat, time_units, scal_range )
%
% Input:
%          DataMat   : a 2D matrix arranged (sample x time), e.g. a row matrix of timeseries.
%          time_units: (optional) integeter specifying the sampling rate
%          scal_range: (optional) 2D vector denoting the [minimum maximum]
%                      timescale interval on which to run DFA. Need to
%                      specify time_units (if time_units=[]), scal_range is
%                      assumed to be in "sampling units"
%
% Output:    
%         out.H_vect    : (sample x 1) vector of Hurst coefficient estimates,
%                         quantifything fractal scaling for the set of samples.
%         out.log_FVAR  : (sample x timescale) matrix of fluctuation log-variance 
%                         as a function of time-scale
%         out.scale_list: (timescale x 1) vector of the corresponding
%                         time-scales (i.e. window sizes). Either in sampling units,
%                         or "time_units" increments, if specified in input
%         out.R2        : (sample x 1) vector measuring goodness of fit for
%                         log-log regression plot, used to estimate Hurst
%         out.scal_err  : (timescale x 1) vector measuring mean error on regression
%                         fit for each timescale

% matrix dmensions
[Nvox Nsamp] = size( DataMat );

if(isempty(varargin))
    time_units = [];
    scal_range = [3 Nsamp];
    disp('running analysis on full range');
elseif( numel(varargin)==1 )
    time_units = varargin{1};
    scal_range = [3 Nsamp];
    disp('running analysis on full range');
elseif( numel(varargin)==2 )
    time_units = varargin{1};
    scal_range = varargin{2};
    
    if( ~isempty(scal_range) && length(scal_range)~=2 )
        error('scal_range must be a 2D vector');
    end
else
    error('number of arguments exceeds possible inputs'); 
end
    
% STEP 0: linear+constant detrending on full data matrix, to minimize
%         any obvious artifact due to scanner drift, etc.
% const+linear regressors
D01 = [ ones(Nsamp,1), linspace(-1,1,Nsamp)' ];
% compute beta (Regression coefficients)
BetaFits = DataMat * (D01/( D01' * D01 )); 
% subtract estimated const+linear
DetrMat  = DataMat - (BetaFits * D01');    

% STEP 1: prepare for multiscale analysis
%
% get the cumulative histogram at each voxel
YPER = cumsum( DetrMat, 2 );

%----------- getting scales -----------%

% rearraning, min to max
scal_range = [min(scal_range) max(scal_range)];
% adjusting bounds if needed
if( scal_range(1) < 3     ) 
    disp('min. range is too small. Adjusting....');
    scal_range(1) = 3;    
end 
if( scal_range(2) > Nsamp )
    disp('max. range is too large. Adjusting....');
    scal_range(2) = Nsamp; 
end 

% finest sampling possible before duplication
KWIND=5;
seg0 = round(exp(linspace( log(scal_range(1)), log(scal_range(2)), KWIND)));
while( (length(unique(seg0)) - length(seg0)) == 0 )
    KWIND=KWIND+1;
    seg0 = round(exp(linspace( log(scal_range(1)), log(scal_range(2)), KWIND)));    
end

seglist = unique(seg0);
klist   = floor(Nsamp./seglist);

%--------------------------------------%

% unit specifications?
if    ( isempty(time_units) )   scale_list = seglist;
elseif( isnumeric(time_units) ) scale_list = seglist .* time_units;
else   error('time_units field must be numeric (or empty)!');
end
% adjusting range?

disp(['...computed over K=',num2str(length(klist)),' timescale points']);

% .initialize variance data matrix
% .then estimate windowed power in 2 configurations (minimize edge effects)
FVAR    = zeros( Nvox, length(klist), 2);

%% 1. front-to-back power estimation
for(k=1:length(klist)) % k indep. splits

    Nelem = seglist(k); % number of time-points per segment
    Nsplt = klist(k);   % number of segments
    %
    % linear-constant regressors
    D01 = [ ones(Nelem,1), linspace(-1,1,Nelem)' ];    

    for(q=1:Nsplt)
        tmp  = YPER(:,(q-1)*Nelem+1:q*Nelem);
        Beta = tmp * (D01/( D01' * D01 )); 
        FVAR(:,k,1) = FVAR(:,k,1) + sum((tmp - (Beta * D01')).^2,2)./( Nsplt*Nelem );
    end    
end
 
%% 2. back-to-front power estimation

YPER = fliplr(YPER); % flipping the integrated timeseries

for(k=1:length(klist)) % k indep. splits

    Nelem = seglist(k); % number of time-points per segment
    Nsplt = klist(k);   % number of segments
    %
    % linear-constant regressors
    D01 = [ ones(Nelem,1), linspace(-1,1,Nelem)' ];    

    for(q=1:Nsplt)
        tmp  = YPER(:,(q-1)*Nelem+1:q*Nelem);
        Beta = tmp * (D01/( D01' * D01 )); 
        FVAR(:,k,2) = FVAR(:,k,2) + sum((tmp - (Beta * D01')).^2,2)./( Nsplt*Nelem );
    end    
end

disp('...done computing power. now calculating Hurst.');

% average both configs., then take the sqrt
FVAR = sqrt(mean(FVAR,3));
% log-transform the RSS variation
log_FVAR = log(FVAR);
% linear fit with log( segment size )
REG = [ ones(length(seglist(:)),1), log(seglist(:)) ];
% Get the Beta -- linearity fit with log( segment size )
BetaFits  = log_FVAR * (REG/( REG' * REG )); 
% final results take only the linear fit (component 2)
H_vect = BetaFits(:,2);

% estimated linear fit
log_FVAR_estim = BetaFits*REG';
% per-scale fit
err = mean( (log_FVAR - log_FVAR_estim).^2 );
% degree of fit
R2 = 1 - sum( (log_FVAR - log_FVAR_estim).^2,2 ) ./ sum( bsxfun(@minus,log_FVAR,mean(log_FVAR,2)).^2, 2 );    

out.H_vect =  H_vect;
out.log_FVAR = log_FVAR;
out.scale_list = scale_list;
out.R2 = R2;
out.scal_err = err;

%%% plotting
% figure,
% subplot(2,2,1); hist( out.H_vect, 0.5:0.025:1 ); xlim([0.5 1.05]);
%                 ylabel('freq'); title('Hurst exponent distro.');
% subplot(2,2,2); hist( out.R2, 0:0.025:1 ); xlim([0 1]);
%                 ylabel('freq'); title('R2 goodness of fit distro.');
% subplot(2,2,3); hold on; errorbar( log(out.scale_list), mean(out.log_FVAR), 1.96*std(out.log_FVAR), '.k');
%                 plot( log(out.scale_list), mean(out.log_FVAR), '.-r'); gx=sort(log(out.scale_list)); xlim([gx(1)-0.1*range(gx) gx(end)+0.1*range(gx)]);
%                 ylabel('log-power'); xlabel('log-scale'); title('mean log-log plot');
% subplot(2,2,4); plot( (out.scale_list), out.scal_err, '.-k' );  gx=sort((out.scale_list)); xlim([gx(1)-0.1*range(gx) gx(end)+0.1*range(gx)]);
%                 xlabel('time scale'); ylabel('sse'); title('error in linear fit');
%                 