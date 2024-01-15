function out = myregplots( x, y, y0 )

    ixkep = isfinite( x ) & isfinite( y);
    ixkep0= isfinite( y0 );
    
    x = x(ixkep);
    y = y(ixkep);
    y0= y0(ixkep0);

    if(~isempty(ixkep0) )
        oo = quickboot( y0 );
    end

    hold on;
    xbsamp = linspace( min(x), max(x), 100 )';
    P = polyfit(x,y,1);
    b0set(1,1) = P(1);
    i0set(1,1) = P(2);
    r0set(1,1) = corr(x,y).^2;
    ybest0 = polyval(P,xbsamp);
    for(bsr=1:1000)
        list= ceil( numel(x) * rand( numel(x),1 ) ); 
        list0=ceil( numel(y0)*rand( numel(y0),1));
        xbb=x(list);
        ybb=y(list);
        if(~isempty(ixkep0) )
            ybbb0=y0(list0);
        end
        P = polyfit(xbb,ybb,1);
        ybest(:,bsr) = polyval(P,xbsamp);
        r2set(bsr,1) = corr( xbb,ybb ).^2;
        b2set(bsr,1) = P(1);
        i2set(bsr,1) = P(2);
        if(~isempty(ixkep0) )
            c2set(bsr,1) = mean(ybbb0);
        end
    end
    shadefill( (xbsamp), prctile(ybest,97.5,2), prctile(ybest,2.5,2), [0.95 0.75 0.75], 500 );
        
    if(~isempty(ixkep0) )
        plot( [min(x) max(x)], oo.av*[1 1], '-k', 'linewidth',3 );
        plot( [min(x) max(x)], oo.ci(1)*[1 1], '--k', 'linewidth',2 );
        plot( [min(x) max(x)], oo.ci(2)*[1 1], '--k', 'linewidth',2 );
    end
    rng = 0.05*[max(y)-min(y)]; ylim( [min(y)-rng, max(y)+rng] );
    rng = 0.05*[max(x)-min(x)]; xlim( [min(x)-rng, max(x)+rng] );
    
    plot( x,y, 'ok', 'markerfacecolor','r', 'markersize',8 ); 
    plot( xbsamp,ybest0,'-','color',[0.7 0.2 0.2],'linewidth',3);

    (b0set),
    mean(b2set)./std(b2set),
    prctile(b2set,[2.5 97.5]),
    (i0set),
    mean(i2set)./std(i2set),
    prctile(i2set,[2.5 97.5]),
    (r0set),
    prctile( r2set, [2.5 97.5]),
    
    out=[];
    
    
    
    
    
    
    