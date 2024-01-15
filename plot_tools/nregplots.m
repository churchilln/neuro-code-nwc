function out = nregplots( x, y, coef_distr, arg1,arg2,arg3  ) %y0, y0_distr
%
% nregplots( x, y, coefs_distro(opt), [control-model]
%
% [control-model] --> ( y0, y0_distro(opt) ) --> y0_distro via (mean LV, UB)
% [control-model] --> ( x0, x0, coefs_distro(opt) ) --> coefs_distro via (resamp x coef-ord)
%
% coeff ordering --> [0th, 1st]
%
marksz = 8;
    
    if(nargin<3) coef_distr=[]; end
    if(nargin<4) arg1=[]; end
    if(nargin<5) arg2=[]; end
    if(nargin<6) arg3=[]; end

    % control things first

    if (isempty(arg2) && isempty(arg3)) || (numel(arg2)==3 && isempty(arg3)) % control == meandiff
        modec = 'mean';
        y0 = arg1;
        y0_distr = arg2;
    
        ixkep = isfinite( x ) & isfinite( y);
        ixkep0= isfinite( y0 );
        
        x = x(ixkep);
        y = y(ixkep);
        y0= y0(ixkep0);
    
        % distributional stats for control
        if(~isempty(ixkep0) )
            if(isempty(y0_distr))
                oo = quickboot( y0 );
                av_ci_ctl = [oo.av oo.ci];
            else
                av_ci_ctl = y0_distr;
            end
        end
        
        % treatment things next
        
        % distributional stats for regression
        if(isempty(coef_distr))
            for(bsr=1:1000)
                list= ceil( numel(x) * rand( numel(x),1 ) ); 
                xbb=x(list);
                xbb=[ones(numel(x),1) xbb];
                ybb=y(list);
                beta_dist(bsr,:) = ybb' * (xbb / (xbb'*xbb));
                cc_dist(bsr,1) = corr( xbb(:,2), ybb );
            end
        else
            beta_dist = coef_distr;
        end
    
        hold on;
    
        xbsamp = linspace( min(x), max(x), 100 )';
        
        if(~isempty(ixkep0) )
            shadefill([min(x) max(x)], av_ci_ctl(3)*[1 1], av_ci_ctl(2)*[1 1], [0.65 0.65 0.65],500);
         end    
        
        y_pred = bsxfun(@plus, (beta_dist(:,2)' .* xbsamp), beta_dist(:,1)' ); % predict-value x boot-model
        y_predU = mean(beta_dist(:,2),1).*xbsamp + mean(beta_dist(:,1));
        shadefill( (xbsamp), prctile(y_pred,97.5,2), prctile(y_pred,2.5,2), [0.95 0.65 0.65], 500 );
            
        rng = 0.10*[max(y)-min(y)]; ylim( [min(y)-rng, max(y)+rng] );
        rng = 0.05*[max(x)-min(x)]; xlim( [min(x)-rng, max(x)+rng] );
    
        if(~isempty(ixkep0) )
            plot( [min(x) max(x)], av_ci_ctl(1)*[1 1], '-k', 'linewidth',3 );
        end    
        
        plot( x,y, 'ok', 'markerfacecolor','r', 'markersize',8 ); 
        plot( xbsamp,y_predU,'-','color',[0.7 0.2 0.2],'linewidth',3);
    
        out=[];
        
        [mean( beta_dist,1 )', prctile( beta_dist, [2.5 97.5] )'],
        [(mean( beta_dist,1 )./std( beta_dist,0,1 ))',  ( 2*min([mean(beta_dist<0,1);mean(beta_dist>0,1)],[],1) )' ],
%         [mean( cc_dist,1 ), prctile( cc_dist, [2.5 97.5] ), 2*min([mean(cc_dist<0), mean(cc_dist>0)])],
      

    elseif numel(arg1) == numel(arg2) % control == regression
        modec = 'reg';
        x0 = arg1;
        y0 = arg2;
        coef_distr0 = arg3;

        ixkep = isfinite( x ) & isfinite( y);
        ixkep0= isfinite( x0 ) & isfinite( y0);
        
        x = x(ixkep);
        y = y(ixkep);
        x0= x0(ixkep0);
        y0= y0(ixkep0);
    
        % distributional stats for ctl-regression
        if(isempty(coef_distr0))
            for(bsr=1:1000)
                list= ceil( numel(x0) * rand( numel(x0),1 ) ); 
                xbb=x0(list);
                xbb=[ones(numel(x0),1) xbb];
                ybb=y0(list);
                beta_dist0(bsr,:) = ybb' * (xbb / (xbb'*xbb));
                cc_dist0(bsr,1) = corr( xbb(:,2), ybb );
            end
        else
            beta_dist0 = coef_distr0;
        end
        
        % distributional stats for regression
        if(isempty(coef_distr))
            for(bsr=1:1000)
                list= ceil( numel(x) * rand( numel(x),1 ) ); 
                xbb=x(list);
                xbb=[ones(numel(x),1) xbb];
                ybb=y(list);
                beta_dist(bsr,:) = ybb' * (xbb / (xbb'*xbb));
                cc_dist(bsr,1) = corr( xbb(:,2), ybb );
            end
        else
            beta_dist = coef_distr;
        end
    
        hold on;
    
        xbsamp = linspace( min(x), max(x), 100 )';
        
        if(~isempty(ixkep0) )
            y_pred0 = bsxfun(@plus, (beta_dist0(:,2)' .* xbsamp), beta_dist0(:,1)' ); % predict-value x boot-model
            y_predU0 = mean(beta_dist0(:,2),1).*xbsamp + mean(beta_dist0(:,1));
            shadefill( (xbsamp), prctile(y_pred0,97.5,2), prctile(y_pred0,2.5,2), [0.65 0.65 0.65], 500 );
        end    
        
        y_pred = bsxfun(@plus, (beta_dist(:,2)' .* xbsamp), beta_dist(:,1)' ); % predict-value x boot-model
        y_predU = mean(beta_dist(:,2),1).*xbsamp + mean(beta_dist(:,1));
        shadefill( (xbsamp), prctile(y_pred,97.5,2), prctile(y_pred,2.5,2), [0.95 0.65 0.65], 500 );
            
        rng = 0.10*[max(y)-min(y)]; ylim( [min(y)-rng, max(y)+rng] );
        rng = 0.05*[max(x)-min(x)]; xlim( [min(x)-rng, max(x)+rng] );
    
        if(~isempty(ixkep0) )
            plot( x0,y0, 'ok', 'markerfacecolor',[0.5 0.5 0.5], 'markersize',8 ); 
            plot( xbsamp,y_predU0,'-','color','k','linewidth',3);
        end    
        
        plot( x,y, 'ok', 'markerfacecolor','r', 'markersize',8 ); 
        plot( xbsamp,y_predU,'-','color',[0.7 0.2 0.2],'linewidth',3);
    
        out=[];
        
        if(~isempty(ixkep0) )
        [mean( beta_dist0,1 )', prctile( beta_dist0, [2.5 97.5] )'],
        [(mean( beta_dist0,1 )./std( beta_dist0,0,1 ))',  ( 2*min([mean(beta_dist0<0,1);mean(beta_dist0>0,1)],[],1) )' ],
        [mean( cc_dist0,1 ), prctile( cc_dist0, [2.5 97.5] ), 2*min([mean(cc_dist0<0), mean(cc_dist0>0)])],
        end

        [mean( beta_dist,1 )', prctile( beta_dist, [2.5 97.5] )'],
        [(mean( beta_dist,1 )./std( beta_dist,0,1 ))',  ( 2*min([mean(beta_dist<0,1);mean(beta_dist>0,1)],[],1) )' ],
        [mean( cc_dist,1 ), prctile( cc_dist, [2.5 97.5] ), 2*min([mean(cc_dist<0), mean(cc_dist>0)])],
      
    else
        error('cannot figure out how to plot your control data~!')
    end

% %     ixkep = isfinite( x ) & isfinite( y);
% %     ixkep0= isfinite( y0 );
% %     
% %     x = x(ixkep);
% %     y = y(ixkep);
% %     y0= y0(ixkep0);
% % 
% %     % distributional stats for control
% %     if(~isempty(ixkep0) )
% %         if(isempty(y0_distr))
% %             oo = quickboot( y0 );
% %             av_ci_ctl = [oo.av oo.ci];
% %         else
% %             av_ci_ctl = y0_distr;
% %         end
% %     end
% %     
% %     % treatment things next
% % 
% %     % distributional stats for regression
% %     if(isempty(coef_distr))
% %         for(bsr=1:1000)
% %             list= ceil( numel(x) * rand( numel(x),1 ) ); 
% %             xbb=x(list);
% %             xbb=[ones(numel(x),1) xbb];
% %             ybb=y(list);
% %             beta_dist(bsr,:) = ybb' * (xbb / (xbb'*xbb));
% %             cc_dist(bsr,1) = corr( xbb(:,2), ybb );
% %         end
% %     else
% %         beta_dist = coef_distr;
% %     end
% % 
% %     hold on;
% % 
% %     xbsamp = linspace( min(x), max(x), 100 )';
% %     
% %     if(~isempty(ixkep0) )
% %         shadefill([min(x) max(x)], av_ci_ctl(3)*[1 1], av_ci_ctl(2)*[1 1], [0.65 0.65 0.65],500);
% %      end    
% %     
% %     y_pred = bsxfun(@plus, (beta_dist(:,2)' .* xbsamp), beta_dist(:,1)' ); % predict-value x boot-model
% %     y_pred0 = mean(beta_dist(:,2),1).*xbsamp + mean(beta_dist(:,1));
% %     shadefill( (xbsamp), prctile(y_pred,97.5,2), prctile(y_pred,2.5,2), [0.95 0.65 0.65], 500 );
% %         
% %     rng = 0.05*[max(y)-min(y)]; ylim( [min(y)-rng, max(y)+rng] );
% %     rng = 0.05*[max(x)-min(x)]; xlim( [min(x)-rng, max(x)+rng] );
% % 
% %     if(~isempty(ixkep0) )
% %         plot( [min(x) max(x)], av_ci_ctl(1)*[1 1], '-k', 'linewidth',3 );
% %     end    
% %     
% %     plot( x,y, 'ok', 'markerfacecolor','r', 'markersize',8 ); 
% %     plot( xbsamp,y_pred0,'-','color',[0.7 0.2 0.2],'linewidth',3);
% % 
% %     out=[];
% %     
% %     [mean( beta_dist,1 )', prctile( beta_dist, [2.5 97.5] )'],
% %     [(mean( beta_dist,1 )./std( beta_dist,0,1 ))',  ( 2*min([mean(beta_dist<0,1);mean(beta_dist>0,1)],[],1) )' ],
% %     [mean( cc_dist,1 ), prctile( cc_dist, [2.5 97.5] ), 2*min([mean(cc_dist<0), mean(cc_dist>0)])],
% %     
% %     
% %     
% %     
% %     
% %     
    