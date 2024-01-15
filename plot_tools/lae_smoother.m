function out = lae_smoother( x, y, hrange, xnew, kern, labels )

% reshaping
x=x(:);
y=y(:);
isort = sortrows([(1:length(x))' x],2);
isort = isort(:,1);
% initialize resampled x
if( isempty( hrange ) && strcmpi(kern,'gauss') )
    h0 = 1.06 * std(x)*(length(x).^-(1/5)); %% silverman rule-of-thumb
    hrange = [h0/10, 10*h0];
end
% initialize resampled x
if( nargin<4 || isempty( xnew ) )
    xnew = linspace( min(x), max(x), 50 );
end
% initialize list of smoothness params to test
hlist = exp( linspace( log(hrange(1)),log(hrange(2)), 100 ) );

%% 1. cross-validation
for(h = 1:length(hlist))
    for(i=1:length(x))

        dist= abs(x-x(i));
        
        if( strcmpi(kern,'box') )
            % x-valid 1
            iko = find( dist<hlist(h) & dist>eps );
            nwt(i,h) = sum( y(iko) )./length(iko);
        elseif( strcmpi(kern,'gauss'))
            % x-valid 2
            wt = exp( -dist.^2/(2*hlist(h)) );
            wt(dist==0)=0; %% drop "training point" from CV
            wt = wt./sum(wt);
            nwt(i,h) = sum( wt.*y );
        end
    end
    cverr(h,1) = mean( (y-nwt(:,h)).^2 );
end

[vh ih] = min( cverr );
disp(['optimal scale is: ',num2str(hlist(ih))]);
h1 = hlist(ih); %% CV-derived optimum

% figure, 
% subplot(2,2,1); semilogx( hlist, cverr,'.-' ); xlim( [hrange]);
% subplot(2,2,2); plot( x,y,'*k', x(isort),nwt(isort,ih),'.-r');

% %% 2. bootstrapped
% for(h=1:length(hlist))
%     for(bsr=1:1000)
% 
%         list= ceil(length(x)*rand(length(x),1));
%         x_bss=x(list);
%         y_bss=y(list);    
% 
%         for(i=1:length(x))
% 
%             dist= abs(x_bss-x(i));
% 
%             if( strcmpi(kern,'box') )
%                 iko = find( dist<hlist(ih) );
%                 nwtB(i,bsr) = sum( y_bss(iko) )./length(iko);
%             elseif( strcmpi(kern,'gauss') )
%                 wt = exp( -dist.^2/(2*hlist(h)) );
%                 wt = wt./sum(wt);
%                 nwtB(i,bsr) = sum( wt.*y_bss );  
%             end
%         end
%     end
%     bsrset(h,:) = [mean(abs(mean(nwtB,2)./std(nwtB,0,2))), median(abs(mean(nwtB,2)./std(nwtB,0,2)))];
% end
% [vh ih] = max( bsrset(:,1) );
% disp(['optimal scale is: ',num2str(hlist(ih))]);
% 
% subplot(2,2,3); semilogx( hlist, bsrset(:,1),'.-' ); xlim( [hrange]);
% subplot(2,2,4); plot( x,y,'*k', x(isort),nwt(isort,ih),'.-r');
% 
% h2 = hlist(ih); %% CV-derived optimum
% 

h2 = hlist(end); %% current just list most flexible model

%% 2. re-estimate mean curve
HHopt = [h0 h1 h2];
figure; 
for( k=1:3 ) 
    
    for(i=1:length(xnew))

        dist= abs(x-xnew(i));

        if( strcmpi(kern,'box') )
            iko = find( dist< HHopt(k) );
            nwt0(i,1) = sum( y(iko) )./length(iko);
        elseif( strcmpi(kern,'gauss') )
            wt = exp( -dist.^2/(2*HHopt(k)) );
            wt = wt./sum(wt);
            nwt0(i,1) = sum( wt.*y );   
        end
    end

    %% 3. bootstrapped error bounds
    for(bsr=1:1000)

        list= ceil(length(x)*rand(length(x),1));
        x_bss=x(list);
        y_bss=y(list);    

        for(i=1:length(xnew))

            dist= abs(x_bss-xnew(i));

            if( strcmpi(kern,'box') )
                iko = find( dist<HHopt(k) );
                nwtB(i,bsr) = sum( y_bss(iko) )./length(iko);
            elseif( strcmpi(kern,'gauss') )
                wt = exp( -dist.^2/(2*HHopt(k)) );
                wt = wt./sum(wt);
                nwtB(i,bsr) = sum( wt.*y_bss ); 
            end
        end
    end

    subplot(1,3,k), hold on;
    plot( x,y, 'ok', 'markerfacecolor',[0.33 0.33 0.33],'markersize',2 );
    plot( xnew, mean(nwtB,2), '-r', 'linewidth', 2 );
    plot( xnew, mean(nwtB,2)+ 1.95*std(nwtB,0,2), '-r', 'linewidth', 0.5 );
    plot( xnew, mean(nwtB,2)- 1.95*std(nwtB,0,2), '-r', 'linewidth', 0.5 );
    plot( [min(x) max(x)], [0 0], '-k' );
    xlabel(labels{1});
    ylabel(labels{2});
    %set(gca,'fontsize',11);

    out=[];
end
