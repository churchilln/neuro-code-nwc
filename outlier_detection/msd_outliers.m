function out=msd_outliers( X, w, symm, robcent )
%
% MSD_OUTLIERS: script to assess influence of datapoints using mean squared
% deviation measure (i.e. frobenius norm) in a sliding time-window
%
% syntax:
%           out=msd_outliers( X, w, symm, robcent )
%
% input:
%           X       : input data matrices (P variables x N samples)
%           w       : width of "base" time window (number of timepoints to compare against)
%           symm    : symmetric ...
%                               0= displacement relative to w preceding timepoints
%                               1= displacement relative to w predence AND w subsequent timepoitns
%           robcent : robust centering ...
%                               0= displacement relative to mean of other timepoints
%                               1= displacement relative to median of other timepoints
%
% output:
%           out.disp      : msd displacement vector
%

dispvct=zeros(size(X,2),1);

if( symm==0 )
      
    for(t=2:size(X,2)-1)
       wind = max([1 t-w]):t-1;
       if( robcent>0 ) Xref=median(X(:,wind),2);
       else Xref = mean(X(:,wind),2);
       end
       dispvct(t,1) = sum( (Xref - X(:,t)).^2 );
    end
    
elseif( symm==1 )
     
    for(t=2:size(X,2)-1)
       wind = [[max([1 t-w]):t-1] , [t+1:min([t+w,size(X,2)])]];
       if( robcent>0 ) Xref=median(X(:,wind),2);
       else Xref = mean(X(:,wind),2);
       end
       dispvct(t,1) = sum( (Xref - X(:,t)).^2 );
    end

end

out.disp = dispvct;
