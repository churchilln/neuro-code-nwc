function shadefill( x, yub, ylb, color, Ninterp, transp )

if nargin<6
    transp=0;
end

x=x(:);
yub=yub(:);
ylb=ylb(:);

xq = linspace( min(x), max(x), Ninterp );

yub_q = interp1(x,yub,xq,'pchip');
ylb_q = interp1(x,ylb,xq,'pchip');

for(i=1:Ninterp)
    if transp>0
        lh=plot( xq(i)*[1 1], [yub_q(i) ylb_q(i)],'-','color',color,'linewidth',2);
        lh.Color = [lh.Color 0.33];
    else
        plot( xq(i)*[1 1], [yub_q(i) ylb_q(i)],'-','color',color,'linewidth',2);
    end
end
