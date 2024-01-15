function out = LBI_compare( out0, out1 )
% 
% runs regressions / group comparisons for LMM-bootstrap-imputed data
% 

if( isempty(out0) )
    
    ns1 = size( out1.LMM_imp.distro_x, 1 );

    for(bsr=1:1000)

        xref1 = bsxfun(@minus, out1.LMM_imp.distro_y(:,2:end,bsr),  out1.LMM_imp.distro_y(:,1,bsr));

        for(t=1:size(xref1,2))
            xbs=[xref1(:,t)];
            ybs=[ones(ns1,1), [out1.LMM_imp.distro_x(:,:,bsr)]];
            btmp = xbs' * (ybs / (ybs'*ybs));
            out.distr_coef(bsr,:,t) = btmp;
        end    
    end
    
else

    ns0 = size( out0.LMM_imp.distro_x, 1 );
    ns1 = size( out1.LMM_imp.distro_x, 1 );

    contr = [zeros(ns0,1); ones(ns1,1)];
    
    for(bsr=1:1000)
        bsr,

        xref0 = bsxfun(@minus, out0.LMM_imp.distro_y(:,2    ,bsr),  out0.LMM_imp.distro_y(:,1,bsr));
        xref1 = bsxfun(@minus, out1.LMM_imp.distro_y(:,2:end,bsr),  out1.LMM_imp.distro_y(:,1,bsr));

        out.distro_unadj_av(bsr,:) = bsxfun(@minus, mean(xref1,1), mean(xref0,1));
        
        for(t=1:size(xref1,2))
            xbs=[xref0; xref1(:,t)];
            ybs=[ones(ns0+ns1,1), contr, [out0.LMM_imp.distro_x(:,:,bsr); out1.LMM_imp.distro_x(:,:,bsr)]];
            btmp = xbs' * (ybs / (ybs'*ybs));
            out.distro_adj_av(bsr,t) = btmp(2);
        end    
    end

    % standard coefficients -- coefficients, distributions, bsrs, p-values
    b_bs = out.distro_unadj_av';
    mattor =[mean(b_bs,2), prctile(b_bs,[2.5 97.5],2), mean(b_bs,2)./std(b_bs,0,2), 2*min( [mean(b_bs<0,2) mean(b_bs>0,2)],[],2)];
    [~,th_fdr]=fdr( mattor(1:end,end),'p',0.05,0);
    out.stat_unadj=[mattor, th_fdr]; % bootstrap params stored
    % standard coefficients -- coefficients, distributions, bsrs, p-values
    b_bs = out.distro_adj_av';
    mattor =[mean(b_bs,2), prctile(b_bs,[2.5 97.5],2), mean(b_bs,2)./std(b_bs,0,2), 2*min( [mean(b_bs<0,2) mean(b_bs>0,2)],[],2)];
    [~,th_fdr]=fdr( mattor(1:end,end),'p',0.05,0);
    out.stat_adj=[mattor, th_fdr]; % bootstrap params stored
end
