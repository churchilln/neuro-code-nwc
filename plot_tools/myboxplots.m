function out = myboxplots( xset, connectlin, xctl )
%
% xset = KxG cell array of values present at each timepoint k=1...K, for groups g=1...G
% xset_distr = Kx3xG matrix of distributional stats (avg, lowerbound, upperbound)
% connectlin = binary, 1=lines connecting means of adjacent timepoints, 0=no lines
% xctl = vector of control values for single timepoint
% xctl_distr = 1x3 vector of distributional stats (avg, lowerbound, upperbound)
%

if(nargin<2) connectlin=[]; end
if(nargin<3) xctl=[]; end

[K] = numel( xset );

hold on;
xcat=[];

if(~isempty(xctl))
   k=0;
   x = xctl; N=numel(x);    
   oo = quickboot( x );
   av_ci = [oo.av oo.ci];
        
   shadefill( linspace(k-0.25, k+0.25,50), repmat(av_ci(2),1,50), repmat(av_ci(3),1,50), [0.75 0.75 0.75], 300 );
   plot( [k-0.25 k+0.25], av_ci(1)*[1 1], '-k', 'linewidth',3.0 );
   plot( [k-0.25 k+0.25], av_ci(2)*[1 1], '-k', 'linewidth',2 );
   plot( [k-0.25 k+0.25], av_ci(3)*[1 1], '-k', 'linewidth',2 );
   plot( [k-0.25 k-0.25], av_ci(2:3),     '-k', 'linewidth',2 );
   plot( [k+0.25 k+0.25], av_ci(2:3),     '-k', 'linewidth',2 );
   plot( randn(N,1)*0.05 + k, x, 'ok', 'markerfacecolor',[0.5 0.5 0.5], 'markersize',8 ); 
   xcat = [xcat; x(:)];
end

for(k=1:K)
   x = xset{k}; N=numel(x);
   oo = quickboot( x );
   av_ci = [oo.av oo.ci];

   shadefill( linspace(k-0.25, k+0.25,50), repmat(av_ci(2),1,50), repmat(av_ci(3),1,50), [0.95 0.75 0.75], 300 );
   plot( [k-0.25 k+0.25], av_ci(1)*[1 1], '-k', 'linewidth',3.0 );
   plot( [k-0.25 k+0.25], av_ci(2)*[1 1], '-k', 'linewidth',2 );
   plot( [k-0.25 k+0.25], av_ci(3)*[1 1], '-k', 'linewidth',2 );
   plot( [k-0.25 k-0.25], av_ci(2:3),     '-k', 'linewidth',2 );
   plot( [k+0.25 k+0.25], av_ci(2:3),     '-k', 'linewidth',2 );
   plot( randn(N,1)*0.05 + k, x, 'ok', 'markerfacecolor','r', 'markersize',8 ); 
   xcat = [xcat; x(:)];
   
   if(~isempty(connectlin))
      if(k>1)
         plot( [k-1, k], [av_prev, av_ci(1)], '-r', 'linewidth',2 );
      end
      av_prev = av_ci(1);
   end
end

rng = 0.05*[max(xcat)-min(xcat)]; ylim( [min(xcat)-rng, max(xcat)+rng] );

if(~isempty(xctl)) xlim([-0.5, K+0.5]);
else               xlim([ 0.5, K+0.5]);
end

out=[];
