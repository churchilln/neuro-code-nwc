function out = myboxplots( xset, xset_distr, connectlin, xctl, xctl_distr, color_set1, color_set2 )
%
% xset = KxG cell array of values present at each timepoint k=1...K, for groups g=1...G
% xset_distr = Kx3xG matrix of distributional stats (avg, lowerbound, upperbound)
% connectlin = binary, 1=lines connecting means of adjacent timepoints, 0=no lines
% xctl = vector of control values for single timepoint
% xctl_distr = 1x3 vector of distributional stats (avg, lowerbound, upperbound)
%
marksz = 5;

if(nargin<2) xset_distr=[]; end
if(nargin<3) connectlin=[]; end
if(nargin<4) xctl=[]; end
if(nargin<5) xctl_distr=[]; end
if(nargin<6) color_set1 = [0.95 0.75 0.75; 0.75 0.75 0.95; 0.95 0.95 0.75; 0.75 0.95 0.75]; end
if(nargin<7) color_set2 = 'rbyg'; end

[K,G] = size( xset );

% if monochrome + multigroup, clone it out to all groups
if numel(color_set2)==1 && G>1
    color_set2 = repmat(color_set2,[1 G]);
end
if size(color_set1,1)==1 && G>1
    color_set1 = repmat(color_set1,[G 1]);
end



if( G>4 ) error('too many groups! Anything more than 3 is going to be cluttered!'); end

hold on;
xcat=[];


if    (G==1) centr=0; width=0.25; jitscal=0.05;
elseif(G==2) centr=[-0.15 0.15]; width = 0.25/2; jitscal=0.035;
elseif(G==3) centr=[-0.20 0 0.20]; width = 0.25/3; jitscal = 0.025;
elseif(G==4) centr=[-0.25 -0.10 0.10 0.25]; width = 0.25/4; jitscal = 0.015;
end

% if only 1 timepoint, blow up spacing a bit
if K==1
    centr = centr*1.75;
end

% control things first

if(~isempty(xctl))
   k=0;
   x = xctl; N=numel(x);
   if(isempty(xctl_distr))
        oo = quickboot( x );
        av_ci = [oo.av oo.ci];
   else        
        av_ci = [xctl_distr];
   end
   shadefill( linspace(k-width, k+width,50), repmat(av_ci(2),1,50), repmat(av_ci(3),1,50), [0.75 0.75 0.75], 300 );
   plot( [k-width k+width], av_ci(1)*[1 1], '-k', 'linewidth',3.0 );
   plot( [k-width k+width], av_ci(2)*[1 1], '-k', 'linewidth',2 );
   plot( [k-width k+width], av_ci(3)*[1 1], '-k', 'linewidth',2 );
   plot( [k-width k-width], av_ci(2:3),     '-k', 'linewidth',2 );
   plot( [k+width k+width], av_ci(2:3),     '-k', 'linewidth',2 );
   plot( randn(N,1)*jitscal + k, x, 'ok', 'markerfacecolor',[0.5 0.5 0.5], 'markersize',marksz ); 
   xcat = [xcat; x(:)];
end

% treatment things nex

for(g=1:G)
for(k=1:K)
   x = xset{k,g}; N=numel(x);
   if(isempty(xset_distr))
        oo = quickboot( x );
        av_ci = [oo.av oo.ci];
   else        
        av_ci = [xset_distr(k,:,g)];
   end
   shadefill( linspace(k-width, k+width,50)+centr(g), repmat(av_ci(2),1,50), repmat(av_ci(3),1,50), color_set1(g,:), 300 );
   plot( [k-width k+width]+centr(g), av_ci(1)*[1 1], '-k', 'linewidth',3.0 );
   plot( [k-width k+width]+centr(g), av_ci(2)*[1 1], '-k', 'linewidth',2 );
   plot( [k-width k+width]+centr(g), av_ci(3)*[1 1], '-k', 'linewidth',2 );
   plot( [k-width k-width]+centr(g), av_ci(2:3),     '-k', 'linewidth',2 );
   plot( [k+width k+width]+centr(g), av_ci(2:3),     '-k', 'linewidth',2 );
   if strcmpi(color_set2(g),'x')
   plot( randn(N,1)*jitscal + k + centr(g), x, 'ok', 'markerfacecolor',[0.5 0.5 0.5], 'markersize',marksz ); 
   else
   plot( randn(N,1)*jitscal + k + centr(g), x, 'ok', 'markerfacecolor',color_set2(g), 'markersize',marksz ); 
   end
   xcat = [xcat; x(:)];
   
   if(~isempty(connectlin)) && connectlin>0
      if(k>1)
         if strcmpi(color_set2(g),'x')
         plot( [k-1, k]+centr(g), [av_prev, av_ci(1)], '-', 'color', [0.5 0.5 0.5], 'linewidth',2 );
         else
         plot( [k-1, k]+centr(g), [av_prev, av_ci(1)], ['-',color_set2(g)], 'linewidth',2 );
         end
      end
      av_prev = av_ci(1);
   end
end
end

if(range(xcat)>eps)
rng = 0.05*[max(xcat)-min(xcat)]; ylim( [min(xcat)-rng, max(xcat)+rng] );
%ylim(1.05*[-max(abs(xcat)) max(abs(xcat))])
end

if(~isempty(xctl)) xlim([-0.5, K+0.5]);
else               xlim([ 0.5, K+0.5]);
end

out=[];
