function out = nregplots( x, ycell, facec, shadec, linec, special_case  ) %y0, y0_distr
%
% assumes x (Nx1), Y (NxS)
%
% facec='r', shadec = [0.95 0.65 0.65], linec=[0.7 0.2 0.2]

if nargin<6
    special_case = 0;
end
if numel(special_case)==1
    special_case = repmat(special_case,numel(ycell),1);
end
    
    hold on;

    G = numel(ycell);

    ycat  =[];
    for g=1:G

        ymat = ycell{g};

        [N,S] = size(ymat);

        y = mean(ymat,2);
    
        % distributional stats for regression
    
        for(bsr=1:1000)
            list= ceil( S * rand( S,1 ) ); 
            xbb=[ones(numel(x),1) x];
            ybb=mean(ymat(:,list),2);
            beta_dist(bsr,:) = ybb' * (xbb / (xbb'*xbb));
            cc_dist(bsr,1) = corr( xbb(:,2), ybb );
        end
        
        xbsamp = linspace( min(x), max(x), 100 )';
    
        y_pred = bsxfun(@plus, (beta_dist(:,2)' .* xbsamp), beta_dist(:,1)' ); % predict-value x boot-model
        y_predU{g} = mean(beta_dist(:,2),1).*xbsamp + mean(beta_dist(:,1));

        if special_case(g)>0
        disp('noshade');
        else
        shadefill( (xbsamp), prctile(y_pred,97.5,2), prctile(y_pred,2.5,2), shadec(g,:), 500 );
        end
            
        ycat = [ycat; mean(y,2)];

        fprintf('group %u:',g);
    
        [mean( beta_dist,1 )', prctile( beta_dist, [2.5 97.5] )'],
        [(mean( beta_dist,1 )./std( beta_dist,0,1 ))',  ( 2*min([mean(beta_dist<0,1);mean(beta_dist>0,1)],[],1) )' ],
        [mean( cc_dist,1 ), prctile( cc_dist, [2.5 97.5] ), 2*min([mean(cc_dist<0), mean(cc_dist>0)])],

    end
    
    for g=1:G

        ymat = ycell{g};

        [N,S] = size(ymat);

        y = mean(ymat,2);
        
        if special_case(g)==1
            plot( xbsamp,y_predU{g},'-','color',linec(g,:),'linewidth',3);
        elseif special_case(g)==2
            plot( xbsamp,y_predU{g},':','color',linec(g,:),'linewidth',3);
        elseif special_case(g)==0
            plot( x,y, 'ok', 'markerfacecolor',facec(g,:), 'markersize',6 ); 
            plot( xbsamp,y_predU{g},'-','color',linec(g,:),'linewidth',3);
        end
    end

    rng = 0.05*[max(ycat)-min(ycat)]; ylim( [min(ycat)-rng, max(ycat)+rng] );
    rng = 0.05*[max(x)-min(x)]; xlim( [min(x)-rng, max(x)+rng] );

    out=[];
    
  
    