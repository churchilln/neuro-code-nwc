function [psdX, f] = pwelch_matrix( X, NFFT, Fs, filter, Nseg, BW )
%
% WELCH POWER ESTIMATION FOR 2D MATRICES
%

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
