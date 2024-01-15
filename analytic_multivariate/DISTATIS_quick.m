function out = DISTATIS_quick( CCmat, type, colorset, typestr, Niter )
%
% =========================================================================
% DISTATIS_QUICK: rapid 3-way multidimensional scaling code, with
% bootstrapped confidence intervals
% =========================================================================
%
% Syntax:
%         out = DISTATIS_quick( CCmat, type, colorset, typestr, Niter )
%
% Input:
%         DISTATIS requires (KxK) matrices of distance/similarity measures
%         (e.g. a correlation matix), between K different conditions.
%         It requires a (KxK) matrix per subject, with at least 5 input
%         subjects, to obtain a stable solution
%         
%          CCmat    : 3D matrix of "stacked" 2D distance/similarity matrices. 
%                     For N subjects, CCmat has dimensions (K x K x N)
%          type     : string specifying what type of measure CCmat contains.
%                       'sim' = similarity matrix (e.g. correlations)
%                       'dist'= distance matrix   (e.g. euclidean distance)
%
%          colorset : a (K x 3) matrix, where the kth row denotes the RGB colour
%                     when plotting condition k (k=1...K). 
%                     If colorset=[], colour values are randomly assigned
%          typestr  : vector of characters, where the kth element determines 
%                     the line formatting for the ellipse of condition k (k=1...K)
%                     If typestr=[], all ellipses have format '-'
%          Niters   : number of bootstrap resamples, used to estimate
%                     components (at least Niters=50 recommended)
%
% Output: 
%          Produces a multidimensional scaling plot of the K 95% CI ellipses
%          of each condition (obtained via Boostrap resampling). This plot
%          shows the first 2PCs of greatest variance in DISTATIS space.
%
%          Non-overlapping ellipses represent significantly different
%          groups; overlapping ellipses are not distinguishable in this
%          space (although they may be significantly different along some
%          other PC dimension).
%
% ------------------------------------------------------------------------%
% Author: Nathan Churchill, University of Toronto
%  email: nathan.churchill@rotman.baycrest.on.ca
% ------------------------------------------------------------------------%
% version history: Jan 15 2014
% ------------------------------------------------------------------------%
%

out     = [];
CI_prob = 0.95; % probability bounds
PCmax   = 2;    % max. number of PCdims to record in resampling

% input matrix dimensions
[K1 K2 N] = size(CCmat);
% catches for bad data formatting
if(K1==K2) K=K1;
else       error('similarity matrices must be square!');
end
if(N<5)    error('not enough subjects (dim3) to do Bootstrap resampling');
end

if( isempty(colorset) ) colorset = rand(K,3); end
if( isempty(typestr)  ) typestr  = repmat( '-', 1,K ); end
if( isempty(Niter)    ) Niter    = 1000; end

%% PREPARING DATA MATRICES

% centring matrix
E = eye(K) - ones(K)./K;
% initialize scp matrix
SCP = zeros( size(CCmat) );
% 
if    ( strcmp(type, 'sim' ) ) for(n=1:N) SCP(:,:,n) = 0.5*E*CCmat(:,:,n)*E; end
elseif( strcmp(type, 'dist') ) for(n=1:N) SCP(:,:,n) =-0.5*E*CCmat(:,:,n)*E; end
else                           error('invalid datatype');
end
% normalize each matrix by first eigenvalue
for(n=1:N)
    if(sum(sum(abs(SCP(:,:,n))))>eps)
    [v l] = svd(SCP(:,:,n)); 
    SCP(:,:,n) = SCP(:,:,n) ./ l(1,1);
    end
end

%% POPULATION COMPROMISE

% initialize RV matrix
RVmat = ones(N);
% populate RV matrix
for(i=1:N-1)
    for(j=i+1:N)
        rv = trace( SCP(:,:,i)*SCP(:,:,j) )./ sqrt( trace( SCP(:,:,i)*SCP(:,:,i) ) * trace( SCP(:,:,j)*SCP(:,:,j) ) );
        rv(~isfinite(rv))=1;
        %
        RVmat(i,j) = rv;
        RVmat(j,i) = rv;
    end
end
% decomp on RV matrix
[p theta] = svd( RVmat );
% comproise weights
alfa = p(:,1) ./ sum(p(:,1));
% compromise SCP
S_plus = zeros(K);
for(n=1:N) S_plus = S_plus + SCP(:,:,n) .* alfa(n); end
%
[V L] = svd( S_plus );
% percentvar
PercLoad = diag(L(1:2,1:2))./trace(L);
% factor loadings
F_plus = V(:,1:PCmax) * sqrt(    L(1:PCmax,1:PCmax) );
RP     = V(:,1:PCmax) * sqrt(inv(L(1:PCmax,1:PCmax)));

%% BOOTSTRAPPED COMPROMISE

% initialize bootstrapped projection score matrix
F_boot = zeros( K, PCmax, Niter );

for( bb=1:Niter )
    
    disp(['boostrap iter:_',num2str(bb),'/',num2str(Niter)]);
    
    % bootstrapped SCP matrix
    list     = ceil(N*rand(N,1));
    SCP_boot = SCP(:,:,list);
    
    % initialize RV matrix
    RVmat = ones(N);
    % populate RV matrix
    for(i=1:N-1)
        for(j=i+1:N)
            rv = trace( SCP_boot(:,:,i)*SCP_boot(:,:,j) )./ sqrt( trace( SCP_boot(:,:,i)*SCP_boot(:,:,i) ) * trace( SCP_boot(:,:,j)*SCP_boot(:,:,j) ) );
            rv(~isfinite(rv))=1;
            %
            RVmat(i,j) = rv;
            RVmat(j,i) = rv;
        end
    end
    % decomp on RV matrix
    [p theta] = svd( RVmat );
    % comproise weights
    alfa = p(:,1) ./ sum(p(:,1));
    % compromise SCP
    S_plus = zeros(K);
    for(n=1:N) S_plus = S_plus + SCP_boot(:,:,n) .* alfa(n); end

    % bootstrapped projection scores
    F_boot(:,:,bb) = S_plus * RP;
end

% dimensions: (boot x 2) x Kgroups
F_boot = permute( F_boot, [3 2 1] );
% set nbound for CIs
Nidx = round( CI_prob*Niter );

%% FIGURE PLOTTING

figure, hold on;
% plotting limits
pc1lim  = max(max(abs(F_boot(:,1,:)),[],3),[],1);
pc2lim  = max(max(abs(F_boot(:,2,:)),[],3),[],1);
pc12lim = max([pc1lim pc2lim]);

% go through and estimate CI parameters
for(k=1:K)
    
    % bootstrapped
    F_temp = F_boot(:,:,k);

    % xy coordinates
    x=F_temp(:,1);
    y=F_temp(:,2);
    % get mean
    xo = mean(x);
    yo = mean(y);
    % center coords
    x = x-xo;
    y = y-yo;

    % pca decomposition
    [u l v] = svd( [x y] );
    % PCspace coordinates, centered
    q  = u * l;
    % mahalanobis distance
    Md = sqrt(sum( q.^2 ./ repmat( var(q), [size(q,1) 1] ), 2));
    % list by increasing MD value
    index  = sortrows( [(1:Niter)' Md], 2 );
    ibound = index(Nidx,1);
    % point on CI boundary in pc-space
    qx_bnd = q(ibound,1);
    qy_bnd = q(ibound,2);
    
    % NB: 'a' = major (x) axis, 'b' = minor (y) axis
    % fractional scaling ratio 'c' of b/a
    c = l(2,2)./l(1,1);
    % scaling on 'a'
    a = sqrt( qx_bnd.^2 + (qy_bnd.^2)./(c.^2) );
    % scaling on 'b'
    b = a.*c;
    
    % angle of major axis (relative to x)
    phi = atan( v(2,1) ./ v(1,1) );
    % degrees in rad.
    t=0:0.01:2*pi;
    % trace of ellipse in (x,y) coordinates
    xe = xo + a*cos(t)*cos(phi) - b*sin(t)*sin(phi);
    ye = yo + a*cos(t)*sin(phi) + b*sin(t)*cos(phi);
    
    plot( xo,yo,'ok', 'markersize',4, 'markerfacecolor', colorset(k,:) ); hold on;
    plot( xe,ye, typestr(k), 'color', colorset(k,:), 'linewidth', 2);
end

plot( 0.9*sqrt(L(1,1))*[-1 1], [0 0], '-k', [0 0], 0.9*sqrt(L(2,2))*[-1 1], '-k' );

text(0.9*pc12lim, 0.1*pc12lim ,['var:',num2str(round(100*PercLoad(1))),'%']);
text(0.1*pc12lim, 0.9*pc12lim ,['var:',num2str(round(100*PercLoad(2))),'%']);

%
xlim([-1.1*pc12lim 1.1*pc12lim]);
ylim([-1.1*pc12lim 1.1*pc12lim]);
