function createellipse( x,y, color, format, SCAL )

ixexcl = ~isfinite( x+y );
x(ixexcl)=[];
y(ixexcl)=[];

x_ref = x;
y_ref = y;

% get mean
xo = mean(x);
yo = mean(y);
% center coords
x = x-xo;
y = y-yo;
% pca
[u l v] = svd( [x y] );
%
phi = atan( v(2,1) ./ v(1,1) );
%
scor_a = [x y] * v(:,1);
scor_b = [x y] * v(:,2);

a = SCAL*1.00*std(scor_a); %1.64
b = SCAL*1.00*std(scor_b);

t=0:0.01:2*pi;

x = xo + a*cos(t)*cos(phi) - b*sin(t)*sin(phi);
y = yo + a*cos(t)*sin(phi) + b*sin(t)*cos(phi);

hold on;

% plot( x_ref,y_ref, '.', 'color',color );
% plot( mean(x),mean(y), '+', 'markersize', 8, 'color', color, 'linewidth',1.5 );
plot( mean(x_ref),mean(y_ref), 'ok', 'markersize', 6, 'markerfacecolor', color, 'linewidth',0.5 );
plot( x,y, format, 'color', color, 'linewidth',2);
