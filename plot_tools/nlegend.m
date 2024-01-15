function out = nlegend( xoffset, yoffset, labels, linecolor, shadecolor, scalwid )

if(nargin<6) scalwid = 0.025; end % default

hu = gca;
hu.XLim;
xmin = hu.XLim(1); 
ymin = hu.YLim(1); 
xrng = hu.XLim(2)-hu.XLim(1);
yrng = hu.YLim(2)-hu.YLim(1);

hold on;
for(g=1:numel(labels))
    xcnt(g) = xmin + 0.8*xrng + xoffset;
    ycnt(g) = ymin + 0.9*yrng + yoffset - (g-1)*(3*scalwid)*yrng;
    
    text(xcnt(g) + (1.5*scalwid)*xrng, ycnt(g), labels{g}, 'FontSize', 15);
    
    shadefill( [-scalwid, scalwid].*xrng + xcnt(g), [ scalwid,  scalwid].*yrng + ycnt(g), [-scalwid, -scalwid].*yrng + ycnt(g), shadecolor(g,:), 500 );
    
    plot( [-scalwid, -scalwid].*xrng + xcnt(g), [-scalwid,  scalwid].*yrng + ycnt(g), ['-',linecolor(g)],'linewidth',2 );
    plot( [ scalwid,  scalwid].*xrng + xcnt(g), [-scalwid,  scalwid].*yrng + ycnt(g), ['-',linecolor(g)],'linewidth',2 );
    plot( [-scalwid,  scalwid].*xrng + xcnt(g), [-scalwid, -scalwid].*yrng + ycnt(g), ['-',linecolor(g)],'linewidth',2 );
    plot( [-scalwid,  scalwid].*xrng + xcnt(g), [ scalwid,  scalwid].*yrng + ycnt(g), ['-',linecolor(g)],'linewidth',2 );
    
end



out=[];