function out = LMM_bootstrap_imput( YT, XT, tnam, vnam, veqn, noboot, nodistro, ttext )
%
% script takes in data and using bootstrap resampling generates 
% repeated-measures bootstrap samples of data with LMM-based imputation
%
% can also "carry along" covariates of interest to get concurrent sampling distros 
%
% SYNTAX:
%
%   out = LMM_bootstrap_imput( YT, XT, tnam, vnam, veqn, noboot, nodistro, ttext )
%
% INPUT:
%
% YT       = subject x time array
% XT       = subject x predictor array
% tnam     = variable names for time points (cell array of strings)
% vnam     = variable names for columns of XT/XC (cell array of strings)
% veqn     = lmm equation for fitting
% noboot   = binary vector, if =1, then skip bootstrapping
% nodistro = binary vector, if =1, then just provide basic summary stats
% ttext    = string vector, if output should be provided
%
% OUTPUT:
%
%  out.LMM.model                   = LinearMixedModel structure used to generate imputed samples
%  out.LMM_imput.stat              = summative stats of ea. timpoint, as row vectors (mean/95CI LB/95CI UB/BSR/p-val/FDR)
%  out.LMM_imput.stat_contr        = summative stats contrasting timpoints 2+ against timepoint 1, as row vectors (mean/95CI LB/95CI UB/BSR/p-val/FDR)
%  out.LMM_imput.regr              = summative stats for regression of "response at each timepoint" on predictors in XT, as rows vectors per covariarte; each slab represents a time point
%  out.LMM_imput.regr_contr        = summative stats for regression of "response contrasting timpoints 2+ against timepoint 1" on predictors in XT, as  rows vectors per covariarte; each slab represents a time point contrast
%  out.LMM_imput.distro_y          = sampleing distro on Y after imputation [subj x time] x resamples 
%  out.LMM_imput.distro_y          = sampleing distro on X after imputation [subj x predictor] x resamples 
%  out.LMM_imput.distro_regr       = sampleing distro on regression coefficients in "regr" after imputation [predictor x resamples] x timepoint 
%  out.LMM_imput.distro_regr_contr = sampleing distro on regression coefficients in "regr_contr" after imputation [predictor x resamples] x  timepoint contrast
%
% * NB: beta stats will have an extra predictor (intercept) in them
%

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

    if(noboot==1) %% case where we don't even want bootstrap (why??)
        return;
    end

    for(bsr=1:1000)
        
        if( mod(bsr,10)==0 ) 
            fprintf('%s - bsr %u/1000\n', ttext, bsr);
        end
       
        listt = ceil( ns*rand(ns,1) );
    
        XTbs_noimp = XT(listt,:);
        YTbs_noimp = YT(listt,:);
        
        if( isempty(XT) )
            xsamp = [];
        else
            xsamp(:,:,bsr) = XTbs_noimp;
        end
        
        if( sum(~isfinite(YTbs_noimp(:)))==0 )
            ysamp(:,:,bsr) = YTbs_noimp;
        else
            %--- imputation

            vmat=[]; % concat time blocks
            for(k=1:nt)
                vmat = [vmat; [YT(listt,k), BLK(listt,:,k), [XT(listt,:) subix(listt)]]];
            end
            vmat=repmat(vmat,10,1); % whole thing x10
            tab_new  = array2table( vmat, 'VariableNames', vnam2);
            valimp = random( out.LMM.model, tab_new );
            valimp = reshape(valimp,ns,[]); % (A,S,R,M,Y)#1, (A,S,R,M,Y)#2 ...
            YTbs_avrnd  = zeros(size(YTbs_noimp));
            for(m=1:10)
                YTbs_avrnd = YTbs_avrnd + valimp(:,(1:nt) + nt*(m-1))./10;
            end      
            YTbs_avrnd( isfinite(YTbs_noimp))=0; % from imputed block, drop entries where value exists 
            YTbs_noimp(~isfinite(YTbs_noimp))=0; % from unimputed block, drop entries where value doesnt exist
            ysamp(:,:,bsr) = YTbs_noimp + YTbs_avrnd; 
        end
        
        xtmp = [ones(ns,1) xsamp(:,:,bsr)];
        bsamp(:,:,bsr)       = ysamp(:,:,bsr)' * (xtmp / (xtmp'*xtmp));
        bsamp_contr(:,:,bsr) = bsxfun(@minus,ysamp(:,2:end,bsr),ysamp(:,1,bsr))' * (xtmp / (xtmp'*xtmp));
        %
        csamp(:,:,bsr) = corr(ysamp(:,:,bsr),xsamp(:,:,bsr),'type','Spearman');
        csamp_contr(:,:,bsr) = corr( bsxfun(@minus,ysamp(:,2:end,bsr),ysamp(:,1,bsr)), xsamp(:,:,bsr),'type','Spearman');
    end    

    %%
        
    % standard params of time points -- coefficients, distributions, bsrs, p-values
    av_bs = permute( mean(ysamp,1), [2 3 1] );
    mattor =[mean(av_bs,2), prctile(av_bs,[2.5 97.5],2), mean(av_bs,2)./std(av_bs,0,2), 2*min( [mean(av_bs<0,2) mean(av_bs>0,2)],[],2)];
    [~,th_fdr]=fdr( mattor(1:end,end),'p',0.05,0); % fdr
    out.LMM_imp.stat=[mattor, th_fdr]; % bootstrap params stored

    % standard params of time contrast -- coefficients, distributions, bsrs, p-values
    av_bs = permute( mean(bsxfun(@minus,ysamp(:,2:end,:),ysamp(:,1,:)),1), [2 3 1] );
    mattor =[mean(av_bs,2), prctile(av_bs,[2.5 97.5],2), mean(av_bs,2)./std(av_bs,0,2), 2*min( [mean(av_bs<0,2) mean(av_bs>0,2)],[],2)];
    [~,th_fdr]=fdr( mattor(1:end,end),'p',0.05,0); % fdr
    out.LMM_imp.stat_contr=[mattor, th_fdr]; % bootstrap params stored

    %%
    
    bsamp = permute(bsamp,[2 3 1]); % covar x bsr x time
    bsamp_contr = permute(bsamp_contr,[2 3 1]);
    
    % regression cross-sectionale
    mattor =cat(2, mean(bsamp,2), prctile(bsamp,[2.5 97.5],2), mean(bsamp,2)./std(bsamp,0,2), 2*min( cat(2,mean(bsamp<0,2), mean(bsamp>0,2)),[],2) );
    clear th_fdr;
    for(t=1:size(mattor,3)) [~,th_fdr(:,1,t)]=fdr( mattor(1:end,end,t),'p',0.05,0); end
    out.LMM_imp.regr=cat(2, mattor, th_fdr); % bootstrap params stored

    % regression cross-sectionale time contrast
    mattor =cat(2, mean(bsamp_contr,2), prctile(bsamp_contr,[2.5 97.5],2), mean(bsamp_contr,2)./std(bsamp_contr,0,2), 2*min( cat(2,mean(bsamp_contr<0,2), mean(bsamp_contr>0,2)),[],2) );
    clear th_fdr;
    for(t=1:size(mattor,3)) [~,th_fdr(:,1,t)]=fdr( mattor(1:end,end,t),'p',0.05,0); end
    out.LMM_imp.regr_contr=cat(2,mattor, th_fdr); % bootstrap params stored

    %%
    
    csamp = permute(csamp,[2 3 1]); % covar x bsr x time
    csamp_contr = permute(csamp_contr,[2 3 1]);
    
    % regression cross-sectionale
    mattor =cat(2, mean(csamp,2), prctile(csamp,[2.5 97.5],2), mean(csamp,2)./std(csamp,0,2), 2*min( cat(2,mean(csamp<0,2), mean(csamp>0,2)),[],2) );
    clear th_fdr;
    for(t=1:size(mattor,3)) [~,th_fdr(:,1,t)]=fdr( mattor(1:end,end,t),'p',0.05,0); end
    out.LMM_imp.corr=cat(2, mattor, th_fdr); % bootstrap params stored

    % regression cross-sectionale time contrast
    mattor =cat(2, mean(csamp_contr,2), prctile(csamp_contr,[2.5 97.5],2), mean(csamp_contr,2)./std(csamp_contr,0,2), 2*min( cat(2,mean(csamp_contr<0,2), mean(csamp_contr>0,2)),[],2) );
    clear th_fdr;
    for(t=1:size(mattor,3)) [~,th_fdr(:,1,t)]=fdr( mattor(1:end,end,t),'p',0.05,0); end
    out.LMM_imp.corr_contr=cat(2,mattor, th_fdr); % bootstrap params stored

    
    if( nodistro == 1 )
        disp('discarding raw distro!');
    else
        out.LMM_imp.distro_y = ysamp;
        out.LMM_imp.distro_x = xsamp;
        %
        out.LMM_imp.distro_regr = bsamp;
        out.LMM_imp.distro_regr_contr = bsamp_contr;
        %
        out.LMM_imp.distro_corr = csamp;
        out.LMM_imp.distro_corr_contr = csamp_contr;
    end
    
    
    
    