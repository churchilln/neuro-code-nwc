function outputs = rCAA( dataMatA, dataMatB, TR, offSet, band_select )
% 
%==========================================================================
%  RCAA: reproducible Canonical Autocorrelations Analysis (CAA), used to 
%  identify a set of orthonormal timeseries that are (a) maximally 
%  autocorrelated in time and (b) highly spatially reproducible. This has 
%  the extra feature that you can search for components that have central 
%  frequency  in a specific frequency band (see details below).
%==========================================================================
%
% SYNTAX:
%
%   outputs = rCAA( dataMatA, dataMatB, TR, offSet, band_select )
%
% INPUT:
%
%   dataMatA,B  = fMRI data matrices of size (voxels x time), the number of
%                 timepoints can vary between splits. Requires 2 runs to 
%                 identify reproducible spatial structure in data
%   TR          = acquisition time intervals (in sec.)
%   offSet      = autocorrelation lag that is optimized in this model 
%                 (integer value >0). Recommended offSet=1 generally works 
%                 well, unless there is a good reason to choose otherwise
%   band_select = 2D vector, specifying frequency range (in Hz) the 
%                 components' central frequency must be, to select in CAA 
%                 model. Format as "band_select = [Low_threshold High_threshold]"
%                 If you don't want to constrain components by frequency, 
%                 set to empty (e.g. band_select=[]).
%
% OUTPUT:
%
%   outputs = structure with the following elements:
%
%             outputs.rep   : spatial reproducibility of CAA components
%             outputs.SPM   : (voxels x components) matrix of Z-scored, reproducible component spatial maps
%             outputs.TsetA : (time x components) matrix of component timecourses in data split A
%             outputs.TsetB : (time x components) matrix of component timecourses in data split B
%
% ------------------------------------------------------------------------%
%   Copyright 2013 Baycrest Centre for Geriatric Care
%
%   This file is part of the PHYCAA+ program. PHYCAA+ is free software: you 
%   can redistribute it and/or modify it under the terms of the GNU Lesser 
%   General Public License as published by the Free Software Foundation, 
%   either version 3 of the License, or (at your option) any later version.
% 
%   PHYCAA+ is distributed in the hope that it will be useful, but WITHOUT 
%   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
%   FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License 
%   for more details.
% 
%   You should have received a copy of the GNU General Lesser Public 
%   License along with PHYCAA+. If not, see <http://www.gnu.org/licenses/>.
% 
%   This code was developed by Nathan Churchill Ph.D., University of Toronto,
%   during his doctoral thesis work. Email: nchurchill@research.baycrest.org
%
%   Any use of this code should cite the following publication:
%      Churchill & Strother (2013). "PHYCAA+: An Optimized, Adaptive Procedure for 
%      Measuring and Controlling Physiological Noise in BOLD fMRI". NeuroImage 82: 306-325
%
% ------------------------------------------------------------------------%
% version history: 2013/07/21
% ------------------------------------------------------------------------%

%% (1) Pre-specifying model parameters

% threshold for significant correlation in CVs
signif_cutoff  = 0.05;
% mean center data mats
dataMatA = dataMatA - repmat( mean(dataMatA ,2), [1 size(dataMatA ,2)] );
dataMatB = dataMatB - repmat( mean(dataMatB ,2), [1 size(dataMatB ,2)] );
% dimensions
[Nvox NtimeA] = size( dataMatA ); 
[Nvox NtimeB] = size( dataMatB ); 
% shortest time length, limits range of PCs
Ntime_min = min([NtimeA NtimeB]);

if( ~isempty( band_select ) )
% predefine Fourier parameters, if non-empty
Fny      = 0.5 * 1/TR;                  % Nyquist frequency (highest lossless frequency)
NFFT     = 2^nextpow2(Ntime_min);       % Next power of 2 from length of time-axis
f        = Fny*linspace(0,1,NFFT/2+1);  % fourier data corresponds to these frequency points
end

%% (2) Iterative estimation of autocorrelated timeseries

% select PC subsets:
% Maximum should be < 50% of time-points (samples) to ensure stable CAA solution    
pcMax = round(Ntime_min/2 - 2);
% initialize data matrices
currMatA=dataMatA;
currMatB=dataMatB;

ww=0;         % outer loop iterations
terminFlag=0; % flag-> terminate when no more components found
nocompFlag=1; % flag-> do not record output if no components found at all
while( (ww<pcMax) & (terminFlag==0) )

    ww=ww+1;
    disp(['estimating component #',num2str(ww),'...']);
    
    % SVD --> PCA space representation
    [VA SA temp] = svd( currMatA' * currMatA );
    QA = VA(:,1:round(Ntime_min/2)) * sqrt( SA(1:round(Ntime_min/2),1:round(Ntime_min/2)) ); 
    % SVD --> PCA space representation
    [VB SB temp] = svd( currMatB' * currMatB );
    QB = VB(:,1:round(Ntime_min/2)) * sqrt( SB(1:round(Ntime_min/2),1:round(Ntime_min/2)) ); 
    
    % initialize reproducibility measure + counting measure
    opt_REP  =-1;
    
    % now iterate through each PC subspace dimensionality
    for(pcs=3:(pcMax+1-ww))

        % (SPLIT 1)
        % estimate temporal autocorrelation maximized "sources"
        Q1 = QA( 1:(end-offSet) , 1:pcs ); % un-offset
        Q2 = QA( (offSet+1):end , 1:pcs ); % offsetted timeseries
        % canonical correlations on time-lagged data
        [L1,L2,R,C1,C2,stats] = canoncorr(Q1,Q2); 
        % getting stable "average" autocorrelated timeseries
        a=[C1(1:offSet,:)]; b=[C1(offSet+1:end,:) + C2(1:end-offSet,:)./2]; c=[C2(end-offSet+1:end,:)]; 
        % normalizing the timeseries to unit variance
        tset = unitnorm_ex([a;b;c],1);
        % keep significantly autocorrelated components
        TsetA  = tset(:, stats.pChisq(:) < signif_cutoff );
        
        if( ~isempty( band_select ) )        
            % If frequency band specified: fast-fourier on data
            tsrFrq     = abs( fft(TsetA, NFFT ,1) ) / Ntime_min;
            % get timeseries and power spectra, normalized:
            tsrFrqNrm  = tsrFrq(1:NFFT/2+1,:) ./ repmat( sum(tsrFrq(1:NFFT/2+1,:)), [NFFT/2+1, 1] );
            % = estimate spectral power expectation value = %
            fract_expect = sum(tsrFrqNrm .* repmat(f(:),[1 size(tsrFrqNrm,2)]));
            % identify components with expect-value in the pre-specified frequency band
            bandIdx      = intersect( find(fract_expect>=band_select(1)), find(fract_expect<=band_select(2)) );
            TsetA        = TsetA(:,bandIdx);
        end
        % get spatial weighting maps
        BsetA  = currMatA * TsetA; 
        
        % (SPLIT 2)
        % estimate temporal autocorrelation maximized "sources"
        Q1 = QB( 1:(end-offSet) , 1:pcs ); % un-offset
        Q2 = QB( (offSet+1):end , 1:pcs ); % offsetted timeseries
        % canonical correlations on time-lagged data
        [L1,L2,R,C1,C2,stats] = canoncorr(Q1,Q2); 
        % getting stable "average" autocorrelated timeseries
        a=[C1(1:offSet,:)]; b=[C1(offSet+1:end,:) + C2(1:end-offSet,:)./2]; c=[C2(end-offSet+1:end,:)]; 
        % normalizing the timeseries to unit variance
        tset = unitnorm_ex([a;b;c],1);
        % keep significantly autocorrelated components
        TsetB  = tset(:, stats.pChisq(:) < signif_cutoff );
        
        if( ~isempty( band_select ) )        
            % If frequency band specified: fast-fourier on data
            tsrFrq     = abs( fft(TsetB, NFFT ,1) ) / Ntime_min;
            % get timeseries and power spectra, normalized:
            tsrFrqNrm  = tsrFrq(1:NFFT/2+1,:) ./ repmat( sum(tsrFrq(1:NFFT/2+1,:)), [NFFT/2+1, 1] );
            % = estimate spectral power expectation value = %
            fract_expect = sum(tsrFrqNrm .* repmat(f(:),[1 size(tsrFrqNrm,2)]));
            % identify components with expect-value in the pre-specified frequency band
            bandIdx      = intersect( find(fract_expect>=band_select(1)), find(fract_expect<=band_select(2)) );
            TsetB        = TsetB(:,bandIdx);
        end
        % get spatial weighting maps
        BsetB  = currMatB * TsetB;

        if( ( size(TsetA,2) >0 ) & ( size(TsetB,2) >0 ) )
            % if components are found in both splits,
            % find the pair that maximize cross-correlation (reproducibility)
            [ia ib rep] = maxcrosscorr( BsetA, BsetB, 1 );            
            % select most reproducible component pair
            BmaxA = BsetA(:,ia);
            BmaxB = BsetB(:,ib) .* sign( corr(BsetB(:,ib),BmaxA) );
            % get reproducibility value and Z-scored SPM
            [new_REP, rspm_new] = get_rSPM_ex( BmaxA,BmaxB, 1);         
        else
           % otherwise set reproducibility to -1
           new_REP=-1; 
        end
        
        % update if this PC space gives greater reproducibility than the last-best
        if( new_REP > opt_REP )            
            %
            opt_REP   = new_REP;     % update component with greatest Corr
            rspm_opt  = rspm_new;    % update rSPM of optimal components
            opt_TsetA = TsetA(:,ia); % update optimal timeseries, split A
            opt_TsetB = TsetB(:,ib); % update optimal timeseries, split B
        end
    end

    if( opt_REP==-1)
        % if no components found in any PC subspace, terminate now
        terminFlag=1;
    else
        % otherwise, record this optimal component
        
        nocompFlag=0; % this means we have found at least 1 component
                        
        SPM_all(:,ww)   = rspm_opt;
        TSET_allA(:,ww) = opt_TsetA;
        TSET_allB(:,ww) = opt_TsetB;
        REP_all(ww,1)   = opt_REP;

        % remove this component from data, and start estimation again
        currMatA = ols_regress_ex( dataMatA, TSET_allA );
        currMatB = ols_regress_ex( dataMatB, TSET_allB );
    end
end

if(nocompFlag==1)
    % if no components found, terminate with a warning
    outputs=[];
    disp('No components found under these constraints!');
else
    % reorder by reproducibility
    idxlist = sortrows( [(1:length(REP_all))' REP_all], -2 );
    idxlist = idxlist(:,1);

    % ============= OUTPUTS ================== %
    outputs.rep    = REP_all(idxlist);     % reproducibility 
    outputs.SPM    = SPM_all(:,idxlist);   % Z-scored map
    outputs.TsetA  = TSET_allA(:,idxlist); % matrix of timecourses, splitA
    outputs.TsetB  = TSET_allB(:,idxlist); % matrix of timecourses, splitB
end

%%
function [ detrVol ] = ols_regress_ex( dataVol, regVcts )
% 
%  OLS regression of timeseries from data matrix
%

% matrix dimensions
[Nmeas Nsamp] = size(dataVol);
% regressors + mean timecourse
X         = [ones(Nsamp,1) regVcts];
% beta map estimation
BetaFits  = inv( X' * X ) * X' * dataVol'; 
% OLS reconstruction of data, based on regressors
detr_est  = ( X * BetaFits )';         
% residual data
detrVol   = dataVol - detr_est;    

%%
function [ Xnorm ] = unitnorm_ex( X, flag )
%
% quick normalization procedure
%

% subtract mean if specified
if(flag>0)
   X = X-repmat( mean(X) , [size(X,1),1] );
end
% normalize to unit variane
norm  = sqrt(sum( X.^2 ));
Xnorm = X./ repmat(norm,[size(X,1),1]);

%%
function [ rep, rSPM ] = get_rSPM_ex( vect1, vect2, keepMean )
%
% get reproducible, Z-scored activation map
%

rep = corr(vect1, vect2);

%(1) getting the mean offsets (normed by SD)
normedMean1 = mean(vect1)./std(vect1);
normedMean2 = mean(vect2)./std(vect2);
%    and rotating means into signal/noise axes
sigMean = (normedMean1 + normedMean2)/sqrt(2);
%noiMean = (normedMean1 - normedMean2)/sqrt(2);
%(2) getting  signal/noise axis projections of (zscored) betamaps
sigProj = ( zscore(vect1) + zscore(vect2) ) / sqrt(2);
noiProj = ( zscore(vect1) - zscore(vect2) ) / sqrt(2);
% noise-axis SD
noiStd = std(noiProj);
%(3) norming by noise SD:
%     ...getting the (re-normed) mean offsets
sigMean = sigMean./noiStd;
%noiMean = noiMean./noiStd; 
%  getting the normed signal/noise projection maps
sigProj = sigProj ./ noiStd;
%noiProj = noiProj ./ noiStd;

% Produce the rSPM:
if    ( keepMean == 1 )   rSPM = sigProj + sigMean;
elseif( keepMean == 0 )   rSPM = sigProj;
end

%%
function [i1 i2 v] = maxcrosscorr( X1, X2, absflag )

N1 = size(X1,2);
N2 = size(X2,2);

cc=corr( X1, X2 );

if(absflag>0)  cc=abs(cc); end

maxc1 = max(cc,[],2); %max corr of each element in X1
maxc2 = max(cc,[],1); %max corr of each element in X2

[v i1] = max( maxc1 );
[v i2] = max( maxc2 );
