function my_error_plot( X, s, colos, type )

[samp dim] = size(X);
% offset for each entry
shift = linspace(0, 0.1, dim);
shift = shift - mean(shift);

hold on;

for(dd=1:dim)
    
    % dotplot
    plot( (1:samp)+shift(dd), X(:,dd), type, 'markersize', 5, 'color', colos(dd), 'linewidth',1.5, 'markerfacecolor', colos(dd) );
    % errorbars
    for(j=1:length(s))
       plot( j*[1 1]+shift(dd), [X(j,dd)+s(j)  X(j,dd)-s(j)], 'color', colos(dd) );
    end
end

xlim([0.5 samp+0.5]);