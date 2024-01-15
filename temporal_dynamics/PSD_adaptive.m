function [out] = PSD_adaptive( DataMat, TR, freq_band, max_wind )
%
% -------------------------------------------------------------------------
% POWER SPECTRAL DENSITY ANALYSIS: very simple, (relatively) computationally
% efficient code for computing the Hurst exponent simultaneously for a
% matrix of time-series vectors. This measures the long-range temporal
% dependency of your signal estimates.
% -------------------------------------------------------------------------
%
% Syntax:
%
%             out = PSD_adaptive( DataMat, time_units, freq_band, max_wind )
%
% Input:
%          DataMat   : a 2D matrix arranged (sample x timepoints), e.g. a row matrix of timeseries.
%          time_units: integeter specifying the sampling rate (required)
%           freq_band: (optional) 2D vector denoting the [minimum maximum] frequency values
%                      if not specified or freq_band=[], full frequency range is analyzed
%            max_wind: (optional) number of frequency bands to average PSD over.
%                      this may improve the robustness of your Hurst
%                      estimates, but check carefully!
%                      max_wind must be <= 2^nextpow2( size(DataMat,2) )
%                      if not specified or freq_band=[], the un-altered PSR is used
%
% Output:    
%         out.H_vect    : (sample x 1) vector of Hurst coefficient estimates,
%                         quantifything fractal scaling for the set of samples.
%         out.log_FVAR  : (sample x frequencies) matrix of fluctuation log-variance 
%                         as a function of frequency
%         out.freq_list: (frequencies x 1) vector of the corresponding frequencies. 
%         out.R2        : (sample x 1) vector measuring goodness of fit for
%                         log-log regression plot, used to estimate Hurst
%         out.scal_err  : (frequencies x 1) vector measuring mean error on regression
%                         fit for each frequency

% matrix dmensions
[Nvox Nsamp] = size( DataMat );

if(nargin<3) freq_band=[]; end
if(nargin<4) max_wind =[]; end

% STEP 0: linear+constant detrending on full data matrix, to minimize
%         any obvious artifact due to scanner drift, etc.
% const+linear regressors
D01 = [ ones(Nsamp,1), linspace(-1,1,Nsamp)' ];
% compute beta (Regression coefficients)
BetaFits = DataMat * (D01/( D01' * D01 )); 
% subtract estimated const+linear
DetrMat  = DataMat - (BetaFits * D01');    

[psdx f] = pwelch_matrix( DetrMat, [], 1/TR, ['hann'], max_wind,[0.02] );

if(isempty(freq_band)) freq_band = [f(2) f(end)]; end

psdx = psdx(:, f>=freq_band(1) & f<=freq_band(2));
f    =     f(  f>=freq_band(1) & f<=freq_band(2));

disp('...done computing power. now calculating Hurst.');

% log-transform the psd
log_FVAR = log(psdx);
% linear fit with log( segment size )
REG = [ ones(length(f(:)),1), log(f(:)) ];
% Get the Beta -- linearity fit with log( segment size )
BetaFits  = log_FVAR * (REG/( REG' * REG )); 
% final results take only the linear fit (component 2)
H_vect = (1-BetaFits(:,2))./2;

% estimated linear fit
log_FVAR_estim = BetaFits*REG';
% per-scale fit
err = mean( (log_FVAR - log_FVAR_estim).^2 );
% degree of fit
R2 = 1 - sum( (log_FVAR - log_FVAR_estim).^2,2 ) ./ sum( bsxfun(@minus,log_FVAR,mean(log_FVAR,2)).^2, 2 );    

out.H_vect =  H_vect;
out.log_FVAR = log_FVAR;
out.freq_list = f;
out.R2 = R2;
out.scal_err = err;


% %%% plotting
% figure,
% subplot(2,2,1); hist( out.H_vect, 0.5:0.025:1 ); xlim([0.5 1.05]);
%                 ylabel('freq'); title('Hurst exponent distro.');
% subplot(2,2,2); hist( out.R2, 0:0.025:1 ); xlim([0 1]);
%                 ylabel('freq'); title('R2 goodness of fit distro.');
% subplot(2,2,3); hold on; errorbar( log(out.freq_list), mean(out.log_FVAR), 1.96*std(out.log_FVAR), '.k');
%                 plot( log(out.freq_list), mean(out.log_FVAR), '.-r'); gx=sort(log(out.freq_list)); xlim([gx(1)-0.1*range(gx) gx(end)+0.1*range(gx)]);
%                 ylabel('log-power'); xlabel('log-freq'); title('mean log-log plot');
% subplot(2,2,4); plot( (out.freq_list), out.scal_err, '.-k' );  gx=sort((out.freq_list)); xlim([gx(1)-0.1*range(gx) gx(end)+0.1*range(gx)]);
%                 xlabel('freq'); ylabel('sse'); title('error in linear fit');

%%
function [psdX, f] = pwelch_matrix( X, NFFT, Fs, filter, Nseg, BW )

if( nargin<6 ) BW=[]; end

%% initial analysis
if( isempty(Nseg) || Nseg==1 )

    if( isempty(NFFT) )
    NFFT         = 2^nextpow2( size(X,2) );
    end

    if( isempty(Fs) )
        Fs = 1;
    end

    if( isempty(filter) )
        w = ones( size(X,2),1 );
    elseif( strcmpi(filter,'hamming') )
        w = hamming( size(X,2) );
    elseif( strcmpi(filter,'hann') )
        w = hann( size(X,2) );
    end


    Xdft = fft( bsxfun(@times,X,w') , NFFT, 2);
    Xdft = Xdft(:, 1:NFFT/2 + 1 );
    psdX = (1/(Fs*size(X,2)*mean(w.^2))) * abs(Xdft).^2;
    psdX(:,2:end-1) = 2*psdX(:,2:end-1);
    
    f = (Fs/2)*linspace(0,1,NFFT/2+1); 
else

    if( isempty(NFFT) )
    NFFT         = 2^nextpow2( Nseg );
    end

    if( isempty(Fs) )
        Fs = 1;
    end

    if( isempty(filter) )
        w = ones( Nseg,1 );
    elseif( strcmpi(filter,'hamming') )
        w = hamming( Nseg );
    elseif( strcmpi(filter,'hann') )
        w = hann( Nseg );
    end

    powSum = 0; nk=0;
    
    for( t0 = 1:round(Nseg/2):(size(X,2)-Nseg+1) ) % step through

        nk=nk+2;
        
        % front-to-back
        tlist = t0:t0+Nseg-1;
        Xsub = X(:,tlist);
        Xdft = fft( bsxfun(@times,Xsub,w') , NFFT, 2);
        Xdft = Xdft(:, 1:NFFT/2 + 1 );
        psdX = (1/(Fs*Nseg*mean(w.^2))) * abs(Xdft).^2;
        psdX(:,2:end-1) = 2*psdX(:,2:end-1);
        % ---
        powSum = powSum + psdX;
        % back-to-front
        tlist = fliplr( size(X,2) - tlist + 1 );
        Xsub = X(:,tlist);
        Xdft = fft( bsxfun(@times,Xsub,w') , NFFT, 2);
        Xdft = Xdft(:, 1:NFFT/2 + 1 );
        psdX = (1/(Fs*Nseg*mean(w.^2))) * abs(Xdft).^2;
        psdX(:,2:end-1) = 2*psdX(:,2:end-1);
        % ---
        powSum = powSum + psdX;        
    end
    
    psdX = powSum ./ nk;
 
    f = (Fs/2)*linspace(0,1,NFFT/2+1); 
end

%% band integration

if(~isempty(BW))
    
    nbw = ceil( Fs/2/BW ); % number of cells
    ilist= zeros(length(f),1);
    
    for(n=1:nbw)
        ilist( (f>((n-1)*BW))  &  (f<=((n)*BW)) ) = n;
    end
    
    f_new = (0:BW:(nbw-1)*BW) + 0.5*BW;
    psdX_new = zeros( size(psdX,1), nbw );
    for(n=1:nbw)
        psdX_new(:,n) = sum( psdX(:,ilist==n),2 );
    end
    %%%%%%%
    %%%%%%%
    psdX = psdX_new;
    f = f_new;
end
