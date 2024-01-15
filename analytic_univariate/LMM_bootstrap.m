function out = LMM_bootstrap( YT, XT, tnam, vnam, veqn, noboot, nodistro, ttext )
%
% script takes in data and using bootstrap resampling generates 
% LMM bootstrapped coefficient estimates and sampling distributions
%
%   out = LMM_bootstrap( YT, XT, tnam, vnam, veqn, noboot, nodistro, ttext )
%
% YT     = subject x time array
% XT     = subject x predictor array
% tnam   = variable names for time points (cell array of strings)
% vnam   = variable names for columns of XT/XC (cell array of strings)
% veqn   = lmm equation for fitting
% noboot = binary vector, if =1, then skip bootstrapping
% noboot = binary vector, if =1, then just provide basic summary stats
% ttext  = string vector, if output should be provided

if(nargin< 6) noboot   = 0; end
if(nargin< 7) nodistro =[]; end
if(nargin< 8) ttext    =[]; end

for(i=1:numel(tnam)) if( strcmpi(tnam{i},'Y'   ) ) error(' variable name Y    is reserved!'); end; end
for(i=1:numel(tnam)) if( strcmpi(tnam{i},'subj') ) error(' variable name subj is reserved'); end; end

[ns nt] = size(YT);

BLK     = zeros( ns,nt,nt );
YVCT    = [];
GMAT    = [];
for(t=1:nt)
   BLK(:,t,t)=1;              % time blocks, stacked
   GMAT = [GMAT; BLK(:,:,t)]; % unfolded time matrix
   YVCT = [YVCT; YT(:,t)];    % unfolded output vector
end

%% (1) %%%% >>>> LMM PARAMETER MODELLING (1) <<<< %%%%

    % setup basic lmm
    vnam2 = [{'Y'},tnam,vnam,{'subj'}];
    subix = (1:ns)';
    vmat  = [YVCT GMAT repmat([XT subix],nt,1) ];
    tab   = array2table( vmat, 'VariableNames',vnam2);
    %---
    out.LMM.model  = fitlme( tab, veqn ); % mixed effect model stored

    if(noboot==1) %% case where we don't even want bootstrap
        return;
    end
    
    % get full-sample fixed effects
    [b0,bn] = fixedEffects(out.LMM.model); 
    out.LMM.bname = bn.Name;
    Dref = designMatrix(out.LMM.model,'Fixed');
    
    % now bootstrapping
    for(bsr=1:1000)
        
       if( mod(bsr,10)==0 ) 
            fprintf('%s - bsr %u/1000\n', ttext, bsr);
       end
       
       idset_bs = subix( ceil( numel(subix)*rand(numel(subix),1) ) );
       tab_bs = tab(1,:);
       Dref_bs = [];
       for(i=1:numel(idset_bs))
           tab_bs  = [tab_bs;   tab( tab.subj == idset_bs(i), : )];
           Dref_bs = [Dref_bs; Dref( tab.subj == idset_bs(i), : )];
       end
       
       % >>>>> stability checking -- re-draw sample if pathological
       cndval = cond( Dref_bs( isfinite(sum(Dref_bs,2)), : ) );
       reit=0;
       while( cndval>500 ) 
           reit=reit+1;
           idset_bs = subix( ceil( numel(subix)*rand(numel(subix),1) ) );
           tab_bs = tab(1,:);
           Dref_bs = [];
           for(i=1:numel(idset_bs))
               tab_bs  = [tab_bs;   tab( tab.subj == idset_bs(i), : )];
               Dref_bs = [Dref_bs; Dref( tab.subj == idset_bs(i), : )];
           end
           cndval = cond( Dref_bs( isfinite(sum(Dref_bs,2)), : ) );
           disp('...rerunning (bad sample)...');
           if(reit>10) error('model is prone to collapse -- bootstrapping not recommended'); end
       end       
       
       tab_bs = tab_bs(2:end,:);
       lmex_bs = fitlme( tab_bs, veqn );
       b_bs(:,bsr) = fixedEffects(lmex_bs);
    end
    
    % standard coefficients -- coefficients, distributions, bsrs, p-values
    mattor =[b0, prctile(b_bs,[2.5 97.5],2), mean(b_bs,2)./std(b_bs,0,2), 2*min( [mean(b_bs<0,2) mean(b_bs>0,2)],[],2)];
    [~,th_fdr]=fdr( mattor(2:end,end),'p',0.05,0); th_fdr = [NaN; th_fdr]; % fdr, sans intercept
    out.LMM.stat=[mattor, th_fdr]; % bootstrap params stored
    
    if( nodistro == 1 )
        disp('discarding raw distro!');
    else
        out.LMM.distro = b_bs';
    end
