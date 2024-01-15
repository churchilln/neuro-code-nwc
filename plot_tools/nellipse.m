function nellipse( xy, es, charstr )

xy( ~isfinite(sum(xy,2)),:) = [];

t=(0:0.01:2*pi)';

E(:,1) =  es./norm(es);
E(:,2) =  E(:,1);
E(2,2) = -E(2,2);

xyp = (xy - mean(xy)) * E;
sss = std(xyp);

xyu = [sss(1)*cos(t), sss(2)*sin(t)];
xyu2= xyu * E';
xyu3= xyu2 + mean(xy);

hold on; plot( xyu3(:,1), xyu3(:,2), charstr );
