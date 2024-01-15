function out = LMM_bootstrap_prototype( YT, YC, XT, XC, tnam, vnam, veqn, cspec, subix, CT, CC, dx, dxc, SubReg, lmmonly, ttext )
%
% script takes in data and using bootstrap resampling generates 
%
% (1) LMM coefficient effects + specialized contrasts
% (2) Simple distributions by time point
% (3) Cross-sectional comparison to controls
%
%   out = LMM_bootstrap_prototype( YT, YC, XT, XC, tnam, vnam, veqn, cspec, subix, CT, CC, dx, dxc )
%
% YT     = subject x time array
% YC     = subject x 1 array (controls)  [opt -> set to empty]
% XT     = subject x predictor array 
% XC     = subject x predictor array (controls)  [opt -> set to empty]
% tnam   = variable names for time points (cell array of strings)
% vnam   = variable names for columns of XT/XC (cell array of strings)
% veqn   = lmm equation for fitting
% cspec  = special contrasts (contr x 2) cell array
% subix  = vector of subject id values (numeric)
% CT     = matrix of covariates to adjust (cross-sectional comparison)
% CC     = matrix of covariates to adjust (cross-sectional; controls)
% dx     = discriminant vector for subgroup analyses (splits on dx<0 vs dx>0)
% dxc    = discriminant vector for controls, if similarly matched
% SubReg = regression vectors, applied only to subset with dx>0 (subject x var -- values for subjects with dx<0 will be discarded)
% lmmonly= binary vector, if =1, then only run lmm steps
%
%
%

if(nargin<14) SubReg=[]; end
if(nargin<15) lmmonly=0; end
if(nargin<16)  ttext=[]; end

for(i=1:numel(tnam)) if( strcmpi(tnam{i},'Y'   ) ) error(' variable name Y    is reserved!'); end; end
for(i=1:numel(tnam)) if( strcmpi(tnam{i},'subj') ) error(' variable name subj is reserved'); end; end

if( ~isempty(dx ) && ( mean(dx >0)<0.2 | mean(dx <0)<0.2 ) ) error('discriminant vector dx unbalanced?'); end
if( ~isempty(dxc) && ( mean(dxc>0)<0.2 | mean(dxc<0)<0.2 ) ) error('discriminant vector dxc unbalanced?'); end

[ns nt] = size(YT);
if(isempty(cspec)) nc=0;
else              [nc  ~] = size(cspec);
end
BLK     = zeros( ns,nt,nt );
YVCT    = [];
GMAT    = [];
for(t=1:nt)
   BLK(:,t,t)=1;              % time blocks, stacked
   GMAT = [GMAT; BLK(:,:,t)]; % unfolded time matrix
   YVCT = [YVCT; YT(:,t)];    % unfolded output vector
end

misstype={'lwise','pwise','imput'};

%% (1) %%%% >>>> LMM PARAMETER MODELLING (1) <<<< %%%%

    vnam2= [{'Y'},tnam,vnam,{'subj'}];
    vmat = [YVCT GMAT repmat([XT subix(:)],nt,1) ];
    tab  = array2table( vmat, 'VariableNames',vnam2);
    idset = subix;
    %---
    out.LMM.model  = fitlme( tab, veqn ); % mixed effect model stored
    
    if(lmmonly==-1) %% case where we don't even want bootstrap
        return;
    end
    
    [b0,bn] = fixedEffects(out.LMM.model); 
    out.LMM.bname = bn.Name;
    Dref = designMatrix(out.LMM.model,'Fixed');
    if(nc>0)
        for(i=1:size(cspec,1))
            cix(i,1)    = find( strcmpi(out.LMM.bname,cspec{i,1}) );
            cix(i,2)    = find( strcmpi(out.LMM.bname,cspec{i,2}) );
            b0_alt(i,1) = b0( cix(i,2) ) - b0( cix(i,1) );
            out.LMM_alt.bname{i,1} = [cspec{i,2},' - ',cspec{i,1}];
        end
    end
    for(bsr=1:1000)
       fprintf('boot-%u,  %s\n',bsr,ttext),
       idset_bs = idset( ceil( numel(idset)*rand(numel(idset),1) ) );
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
           idset_bs = idset( ceil( numel(idset)*rand(numel(idset),1) ) );
           tab_bs = tab(1,:);
           Dref_bs = [];
           for(i=1:numel(idset_bs))
               tab_bs  = [tab_bs;   tab( tab.subj == idset_bs(i), : )];
               Dref_bs = [Dref_bs; Dref( tab.subj == idset_bs(i), : )];
           end
           cndval = cond( Dref_bs( isfinite(sum(Dref_bs,2)), : ) );
           disp('...rerun (bad sample)...');
           if(reit>10) error('model prone to collapse -- bootstrapping not recommended'); end
       end       
       
       tab_bs = tab_bs(2:end,:);
       lmex_bs = fitlme( tab_bs, veqn );
       b_bs(:,bsr) = fixedEffects(lmex_bs);
       
       if(nc>0)
       for(i=1:nc) b_bs_alt(i,bsr) = b_bs( cix(i,2),bsr ) - b_bs( cix(i,1),bsr );
       end
       end
    end
    
    % standard coefficients -- coefficients, distributions, bsrs, p-values
    mattor =[b0, prctile(b_bs,[2.5 97.5],2), mean(b_bs,2)./std(b_bs,0,2), 2*min( [mean(b_bs<0,2) mean(b_bs>0,2)],[],2)];
    [~,th_fdr]=fdr( mattor(2:end,end),'p',0.05,0); th_fdr = [NaN; th_fdr]; % fdr, sans intercept
    out.LMM.stat=[mattor, th_fdr]; % bootstrap params stored
    
    if(nc>0)
    % contrast coefficients -- coefficients, distributions, bsrs, p-values
    mattor =[b0_alt, prctile(b_bs_alt,[2.5 97.5],2), mean(b_bs_alt,2)./std(b_bs_alt,0,2), 2*min( [mean(b_bs_alt<0,2) mean(b_bs_alt>0,2)],[],2)];
    [~,th_fdr]=fdr( mattor(:,end),'p',0.05,0);
    out.LMM_alt.stat=[mattor, th_fdr]; % bootstrap params stored
    end

if(lmmonly==1)
    return;
else 
    
%% (2) %%%% >>>> SIMPLE LONGITUDINAL MODEL (2) <<<< %%%%

    % distributional statistics.....
    clear estim_tmp estim_tmp1 estim_tmp2;
    for(bsr=1:1000)
        bsr,
        listt = ceil( size(YT,1)*rand(size(YT,1),1) );
        YTbs  = YT(listt,:);
        
        %--- listwise
        bs_timeav(bsr,:,1) = mean(YTbs( isfinite(sum(YTbs,2)),: ),1);
        bs_timedf(bsr,:,1) = mean(YTbs( isfinite(sum(YTbs,2)),2:end ) - YTbs( isfinite(sum(YTbs,2)),1 ),1);
        %--- pairwise
        bs_timeav(bsr,:,2) = mean(YTbs,1,'omitnan');
        bs_timedf(bsr,:,2) = mean(YTbs(:,2:end)-YTbs(:,1),1,'omitnan');
        %--- imputation
        vmat=[]; % concat time blocks
        for(k=1:nt)
            vmat = [vmat; [YT(listt,k), BLK(listt,:,k), [XT(listt,:) subix(listt)]]];
        end
        vmat=repmat(vmat,10,1); % whole thing x10
        tab_new  = array2table( vmat, 'VariableNames', vnam2);
        valimp = random( out.LMM.model, tab_new );
        valimp = reshape(valimp,numel(listt),[]); % (A,S,R,M,Y)#1, (A,S,R,M,Y)#2 ...
        for(m=1:10)
            valimp_sub = valimp(:,(1:nt) + nt*(m-1));
            Ytmp       = YTbs; Ytmp(~isfinite(Ytmp))=0;
            Ybs_imp    = Ytmp + valimp_sub .* double( ~isfinite(YTbs) );
            estim_tmp1(m,:) = mean(Ybs_imp,1);
            estim_tmp2(m,:) = mean(Ybs_imp(:,2:end)-Ybs_imp(:,1),1);
        end
        bs_timeav(bsr,:,3) = mean(estim_tmp1,1);
        bs_timedf(bsr,:,3) = mean(estim_tmp2,1);
    end
    % now convert to summary stats
    for(w=1:3)
        difb = bs_timeav(:,:,w); % --> the means
        effix= isfinite(sum(difb,2));
        difb = difb(effix,:); size(difb),
        out.lng_avg.(['stat_',misstype{w}]) = [mean(difb,1)', prctile(difb,[2.5 97.5])', repmat([NaN, NaN, NaN],size(difb,2),1)];
        %
        difb = bs_timedf(:,:,w); % --> the differences
        effix= isfinite(sum(difb,2));
        difb = difb(effix,:); size(difb),
        mattor = [mean(difb,1)', prctile(difb,[2.5 97.5])', (mean(difb)./std(difb))', (2*min( [mean(difb>0,1); mean(difb<0,1)],[],1 ))'];
        [~,th]=fdr( mattor(:,end),'p',0.05,0);
        out.lng_dif.(['stat_',misstype{w}]) = [mattor, th];
    end
    
%% (3) %%%% >>>> CROSS-SECTIONAL COMPARISONS (3) <<<< %%%%

    if(~isempty(YC))
        
        % distributional statistics.....
        clear estim_tmp estim_tmp1 estim_tmp2;
        for(bsr=1:1000)
            bsr,
            listt= ceil( size(YT,1)*rand(size(YT,1),1) );
            listc= ceil( size(YC,1)*rand(size(YC,1),1) ); 
            YTbs =YT( listt,:);
            YCbs =YC(listc,:); 

            if(isempty(CT)) % -------------------------------------- unadjusted

                %--- listwise
                bs_csect(bsr,:,1) = ( mean(YTbs(isfinite(sum(YTbs,2)),:),1) - mean(YCbs,1) );
                %--- pairwise
                bs_csect(bsr,:,2) = ( mean(YTbs,1,'omitnan') - mean(YCbs,1) );
                %--- imputation
                vmat=[]; % concat time blocks
                for(k=1:nt)
                    vmat = [vmat; [YT(listt,k), BLK(listt,:,k), [XT(listt,:) subix(listt)]]];
                end
                vmat=repmat(vmat,10,1); % whole thing x10
                tab_new  = array2table( vmat, 'VariableNames',vnam2);
                valimp = random( out.LMM.model, tab_new ); %imput data, then rearrange as nsubj x sample/replicate
                valimp = reshape(valimp,numel(listt),[]); % (A,S,R,M,Y)#1, (A,S,R,M,Y)#2 ...
                for(m=1:10)
                    valimp_sub      = valimp(:,(1:nt) + nt*(m-1)); % take each successive block of 5 subjxtimepoint
                    Ytmp            = YTbs; Ytmp(~isfinite(Ytmp))=0;
                    YTbs_imp        = Ytmp + valimp_sub .* double( ~isfinite(YTbs) );
                    %
                    estim_tmp(m,:) = mean(YTbs_imp,1) - mean(YCbs,1);
                end
                bs_csect(bsr,:,3) = mean(estim_tmp,1); % averaged imputed estimate

            else % ----------------------------------------- covariate adjusted

                CTbs =CT( listt,:);
                CCbs =CC( listc,:); 

                xall = [[ones(size(CTbs,1),1) CTbs ones(size(CTbs,1),1)]; [ones(size(CCbs,1),1) CCbs zeros(size(CCbs,1),1)]];
                CAT  = [YTbs; repmat(YCbs,1,nt)];
                for(k=1:nt)
                    %--- listwise
                    btmp = CAT(isfinite(sum(CAT,2)),k)' * (xall(isfinite(sum(CAT,2)),:) /( xall(isfinite(sum(CAT,2)),:)'*xall(isfinite(sum(CAT,2)),:) ));
                    bs_csect(bsr,k,1) = btmp(end);
                    %--- pairwise
                    btmp = CAT(isfinite(CAT(:,k)),k)' * xall(isfinite(CAT(:,k)),:) /( xall(isfinite(CAT(:,k)),:)'*xall(isfinite(CAT(:,k)),:) );
                    bs_csect(bsr,k,2) = btmp(end);
                end

                %--- imputation
                vmat=[]; % concat time blocks
                for(k=1:nt)
                    vmat = [vmat; [YT(listt,k), BLK(listt,:,k), [XT(listt,:) subix(listt)]]];
                end
                vmat=repmat(vmat,10,1); % whole thing x10
                tab_new  = array2table( vmat, 'VariableNames',vnam2);
                valimp = random( out.LMM.model, tab_new ); %imput data, then rearrange as nsubj x sample/replicate
                valimp = reshape(valimp,numel(listt),[]); % (A,S,R,M,Y)#1, (A,S,R,M,Y)#2 ...
                for(m=1:10)
                    valimp_sub      = valimp(:,(1:nt) + nt*(m-1)); % take each successive block of 5 subjxtimepoint
                    Ytmp            = YTbs; Ytmp(~isfinite(Ytmp))=0;
                    YTbs_imp        = Ytmp + valimp_sub .* double( ~isfinite(YTbs) );
                    %
                    xall = [[ones(size(CTbs,1),1) CTbs ones(size(CTbs,1),1)]; [ones(size(CCbs,1),1) CCbs zeros(size(CCbs,1),1)]];
                    CAT  = [YTbs_imp; repmat(YCbs,1,nt)];
                    btmp = CAT' * xall /( xall'*xall );
                    estim_tmp(m,:) = btmp(:,end);
                end
                bs_csect(bsr,:,3) = mean(estim_tmp,1); % averaged imputed estimate
            end
        end
        % now convert to summary stats
        for(w=1:3)
            difb = bs_csect(:,:,w);
            effix= isfinite(sum(difb,2));
            difb = difb(effix,:); size(difb),
            mattor = [mean(difb,1)', prctile(difb,[2.5 97.5])', (mean(difb)./std(difb))', (2*min( [mean(difb>0,1); mean(difb<0,1)],[],1 ))'];
            [~,th]=fdr( mattor(:,end),'p',0.05,0);
            out.xct_dif.(['stat_',misstype{w}]) = [mattor, th];
        end
    
    end
if( ~isempty(dx) ) %------------------------> IN CASES OF SUBGROUPS <------
    
%% (4) %%%% >>>> SIMPLE LONGITUDINAL MODEL, SUBGROUPS (4) <<<< %%%%

    % distributional statistics.....
    clear estim_tmp estim_tmp1 estim_tmp2;
    for(bsr=1:1000)
        bsr,
        listt = ceil( size(YT,1)*rand(size(YT,1),1) );
        YTbs  = YT(listt,:);
        dxbs  = dx(listt);
        
        %--- listwise (N,P)
        bs_timeav_N(bsr,:,1) = mean(YTbs( isfinite(sum(YTbs,2)) & dxbs<-eps,: ),1);
        bs_timedf_N(bsr,:,1) = mean(YTbs( isfinite(sum(YTbs,2)) & dxbs<-eps,2:end ) - YTbs( isfinite(sum(YTbs,2)) & dxbs<-eps,1 ),1);
        bs_timeav_P(bsr,:,1) = mean(YTbs( isfinite(sum(YTbs,2)) & dxbs> eps,: ),1);
        bs_timedf_P(bsr,:,1) = mean(YTbs( isfinite(sum(YTbs,2)) & dxbs> eps,2:end ) - YTbs( isfinite(sum(YTbs,2)) & dxbs> eps,1 ),1);
        %--- pairwise (N,P)
        bs_timeav_N(bsr,:,2) = mean(YTbs(dxbs<-eps,:),1,'omitnan');
        bs_timedf_N(bsr,:,2) = mean(YTbs(dxbs<-eps,2:end)-YTbs(dxbs<-eps,1),1,'omitnan');
        bs_timeav_P(bsr,:,2) = mean(YTbs(dxbs> eps,:),1,'omitnan');
        bs_timedf_P(bsr,:,2) = mean(YTbs(dxbs> eps,2:end)-YTbs(dxbs> eps,1),1,'omitnan');
        %--- imputation (N,P)
        vmat=[]; % concat time blocks
        for(k=1:nt)
            vmat = [vmat; [YT(listt,k), BLK(listt,:,k), [XT(listt,:) subix(listt)]]];
        end
        vmat=repmat(vmat,10,1); % whole thing x10
        tab_new  = array2table( vmat, 'VariableNames', vnam2);
        valimp = random( out.LMM.model, tab_new );
        valimp = reshape(valimp,numel(listt),[]); % (A,S,R,M,Y)#1, (A,S,R,M,Y)#2 ...
        for(m=1:10)
            valimp_sub = valimp(:,(1:nt) + nt*(m-1));
            Ytmp       = YTbs; Ytmp(~isfinite(Ytmp))=0;
            Ybs_imp    = Ytmp + valimp_sub .* double( ~isfinite(YTbs) );
            estim_tmp1_N(m,:) = mean(Ybs_imp(dxbs<-eps,:),1);
            estim_tmp2_N(m,:) = mean(Ybs_imp(dxbs<-eps,2:end)-Ybs_imp(dxbs<-eps,1),1);
            estim_tmp1_P(m,:) = mean(Ybs_imp(dxbs> eps,:),1);
            estim_tmp2_P(m,:) = mean(Ybs_imp(dxbs> eps,2:end)-Ybs_imp(dxbs> eps,1),1);
        end
        bs_timeav_N(bsr,:,3) = mean(estim_tmp1_N,1);
        bs_timedf_N(bsr,:,3) = mean(estim_tmp2_N,1);
        bs_timeav_P(bsr,:,3) = mean(estim_tmp1_P,1);
        bs_timedf_P(bsr,:,3) = mean(estim_tmp2_P,1);
    end
    % (N) now convert to summary stats
    for(w=1:3)
        difb = bs_timeav_N(:,:,w); % --> the means
        effix= isfinite(sum(difb,2));
        difb = difb(effix,:); size(difb),
        out.lng_avg_N.(['stat_',misstype{w}]) = [mean(difb,1)', prctile(difb,[2.5 97.5])', repmat([NaN, NaN, NaN],size(difb,2),1)];
        %
        difb = bs_timedf_N(:,:,w); % --> the differences
        effix= isfinite(sum(difb,2));
        difb = difb(effix,:); size(difb),
        mattor = [mean(difb,1)', prctile(difb,[2.5 97.5])', (mean(difb)./std(difb))', (2*min( [mean(difb>0,1); mean(difb<0,1)],[],1 ))'];
        [~,th]=fdr( mattor(:,end),'p',0.05,0);
        out.lng_dif_N.(['stat_',misstype{w}]) = [mattor, th];
    end
    % (P) now convert to summary stats
    for(w=1:3)
        difb = bs_timeav_P(:,:,w); % --> the means
        effix= isfinite(sum(difb,2));
        difb = difb(effix,:); size(difb),
        out.lng_avg_P.(['stat_',misstype{w}]) = [mean(difb,1)', prctile(difb,[2.5 97.5])', repmat([NaN, NaN, NaN],size(difb,2),1)];
        %
        difb = bs_timedf_P(:,:,w); % --> the differences
        effix= isfinite(sum(difb,2));
        difb = difb(effix,:); size(difb),
        mattor = [mean(difb,1)', prctile(difb,[2.5 97.5])', (mean(difb)./std(difb))', (2*min( [mean(difb>0,1); mean(difb<0,1)],[],1 ))'];
        [~,th]=fdr( mattor(:,end),'p',0.05,0);
        out.lng_dif_P.(['stat_',misstype{w}]) = [mattor, th];
    end


%% (4.5) %%%% >>>> SUBSET REGRESSION MODEL, SUBGROUPS (4.5) <<<< %%%%

    rflag = 1; %1=regress, 0=corr

    if(~isempty(SubReg)) % ---> repurposes code from block (4), doing regression only on dx>0 subgroup
        % distributional statistics.....
        clear estim_tmp estim_tmp1 estim_tmp2 estim_tmp1_P;
        for(bsr=1:1000)
            bsr,
            listt = ceil( size(YT,1)*rand(size(YT,1),1) );
            YTbs  = YT(listt,:);
            dxbs  = dx(listt);
            SRbs  = SubReg(listt,:);
            for(j=1:size(SRbs,2)) % median imputation
                tmp=SRbs(:,j); 
                tmp(~isfinite(tmp))=median(tmp(isfinite(tmp)));
                SRbs(:,j)=tmp;
            end

            %--- listwise (N,P)
            for(k=1:nt)
                xss = [ones(sum(isfinite(sum(YTbs,2)) & dxbs> eps),1), SRbs( isfinite(sum(YTbs,2)) & dxbs> eps,: )];
                if(rflag==1)
                    btmp = YTbs( isfinite(sum(YTbs,2)) & dxbs> eps,k )' * (xss /( xss'*xss ));
                else
                    if( sum(isfinite(sum(YTbs,2)) & dxbs> eps)<=1 )
                    btmp = NaN*ones( 1, size(xss,2) );    
                    else
                    btmp = corr(  YTbs( isfinite(sum(YTbs,2)) & dxbs> eps,k ), xss, 'type','Spearman' );
                    end
                end
                bs_subreg_P(bsr,:,k,1) = btmp(2:end);
            end
            %--- pairwise (N,P)
            for(k=1:nt)
                xss = [ones(sum(isfinite(YTbs(:,k)) & dxbs> eps),1), SRbs( isfinite(YTbs(:,k)) & dxbs> eps,: )];
                if(rflag==1)
                    btmp = YTbs( isfinite(YTbs(:,k)) & dxbs> eps,k )' * (xss /( xss'*xss ));
                else
                    btmp = corr(  YTbs( isfinite(YTbs(:,k)) & dxbs> eps,k ), xss, 'type','Spearman' );
                end
                bs_subreg_P(bsr,:,k,2) = btmp(2:end);
            end
            %--- imputation (N,P)
            vmat=[]; % concat time blocks
            for(k=1:nt)
                vmat = [vmat; [YT(listt,k), BLK(listt,:,k), [XT(listt,:) subix(listt)]]];
            end
            vmat=repmat(vmat,10,1); % whole thing x10
            tab_new  = array2table( vmat, 'VariableNames', vnam2);
            valimp = random( out.LMM.model, tab_new );
            valimp = reshape(valimp,numel(listt),[]); % (A,S,R,M,Y)#1, (A,S,R,M,Y)#2 ...
            for(m=1:10)
                valimp_sub = valimp(:,(1:nt) + nt*(m-1));
                Ytmp       = YTbs; Ytmp(~isfinite(Ytmp))=0;
                Ybs_imp    = Ytmp + valimp_sub .* double( ~isfinite(YTbs) );
                xss = [ones(sum(dxbs> eps),1), SRbs( dxbs> eps,: )];
                if(rflag==1)
                    estim_tmp1_P(:,:,m) = Ybs_imp( dxbs> eps,: )' * (xss /( xss'*xss )); % tptxbeh
                else
                    estim_tmp1_P(:,:,m) = corr( Ybs_imp( dxbs> eps,: ), xss, 'type','Spearman' ); % tptxbeh
                end
            end
            bs_subreg_P(bsr,:,:,3) = permute( mean(estim_tmp1_P(:,2:end,:),3), [3 2 1]);
        end
        
        % (P) now convert to summary stats
        for(w=1:3)
            difb = bs_subreg_P(:,:,:,w); % --> the means
            effix= isfinite(sum(sum(difb,2),3));
            difb = difb(effix,:,:); size(difb), %bsr x beh x tpt
            for(j=1:size(bs_subreg_P,3))
                out.lng_subreg_P{j}.(['stat_',misstype{w}]) = [mean(difb(:,:,j),1)', prctile(difb(:,:,j),[2.5 97.5])', (mean(difb(:,:,j))./std(difb(:,:,j)))', (2*min( [mean(difb(:,:,j)>0,1); mean(difb(:,:,j)<0,1)],[],1 ))'];
            end
        end
    end

   return; 
%% (5) %%%% >>>> CROSS-SECTIONAL COMPARISONS, SUBGROUPS (5) <<<< %%%%

    if(~isempty(YC))

        % distributional statistics.....
        clear estim_tmp estim_tmp1 estim_tmp2;
        for(bsr=1:1000)
            bsr,
            listt= ceil( size(YT,1)*rand(size(YT,1),1) );
            listc= ceil( size(YC,1)*rand(size(YC,1),1) ); 
            YTbs =YT( listt,:);
            YCbs =YC(listc,:); 
            dxbs =dx(listt);

    % =================== if no subgrouping imposed on ctls ================= %      
            if(isempty(dxc)) 

                if(isempty(CT)) % -------------------------------------- unadjusted

                    %--- listwise (N,P)
                    bs_csect_N(bsr,:,1) = ( mean(YTbs(isfinite(sum(YTbs,2)) & dxbs<-eps,:),1) - mean(YCbs,1) );
                    bs_csect_P(bsr,:,1) = ( mean(YTbs(isfinite(sum(YTbs,2)) & dxbs> eps,:),1) - mean(YCbs,1) );
                    %--- pairwise (N,P)
                    bs_csect_N(bsr,:,2) = ( mean(YTbs(dxbs<-eps,:),1,'omitnan') - mean(YCbs,1) );
                    bs_csect_P(bsr,:,2) = ( mean(YTbs(dxbs> eps,:),1,'omitnan') - mean(YCbs,1) );
                    %--- imputed (N,P)
                    vmat=[]; % concat time blocks
                    for(k=1:nt)
                        vmat = [vmat; [YT(listt,k), BLK(listt,:,k), [XT(listt,:) subix(listt)]]];
                    end
                    vmat=repmat(vmat,10,1); % whole thing x10
                    tab_new  = array2table( vmat, 'VariableNames',vnam2);
                    valimp = random( out.LMM.model, tab_new ); %imput data, then rearrange as nsubj x sample/replicate
                    valimp = reshape(valimp,numel(listt),[]); % (A,S,R,M,Y)#1, (A,S,R,M,Y)#2 ...
                    for(m=1:10)
                        valimp_sub      = valimp(:,(1:nt) + nt*(m-1)); % take each successive block of 5 subjxtimepoint
                        Ytmp            = YTbs; Ytmp(~isfinite(Ytmp))=0;
                        YTbs_imp        = Ytmp + valimp_sub .* double( ~isfinite(YTbs) );
                        %
                        estim_tmp_N(m,:) = mean(YTbs_imp(dxbs<-eps,:),1) - mean(YCbs,1);
                        estim_tmp_P(m,:) = mean(YTbs_imp(dxbs> eps,:),1) - mean(YCbs,1);
                    end
                    bs_csect_N(bsr,:,3) = mean(estim_tmp_N,1); % averaged imputed estimate
                    bs_csect_P(bsr,:,3) = mean(estim_tmp_P,1); % averaged imputed estimate

                else % ----------------------------------------- covariate adjusted

                    % covariate matrices
                    CTbs =CT( listt,:);
                    CCbs =CC( listc,:); 

                    xall = [[ones(sum(dxbs<-eps),1) CTbs(dxbs<-eps,:) ones(sum(dxbs<-eps),1)]; [ones(size(CCbs,1),1) CCbs zeros(size(CCbs,1),1)]];
                    CAT  = [YTbs(dxbs<-eps,:); repmat(YCbs,1,nt)];
                    for(k=1:nt)
                        %--- listwise (N)
                        btmp = CAT(isfinite(sum(CAT,2)),k)' * (xall(isfinite(sum(CAT,2)),:) /( xall(isfinite(sum(CAT,2)),:)'*xall(isfinite(sum(CAT,2)),:) ));
                        bs_csect_N(bsr,k,1) = btmp(end);
                        %--- pairwise (N)
                        btmp = CAT(isfinite(CAT(:,k)),k)' * xall(isfinite(CAT(:,k)),:) /( xall(isfinite(CAT(:,k)),:)'*xall(isfinite(CAT(:,k)),:) );
                        bs_csect_N(bsr,k,2) = btmp(end);
                    end
                    %
                    xall = [[ones(sum(dxbs> eps),1) CTbs(dxbs> eps,:) ones(sum(dxbs> eps),1)]; [ones(size(CCbs,1),1) CCbs zeros(size(CCbs,1),1)]];
                    CAT  = [YTbs(dxbs> eps,:); repmat(YCbs,1,nt)];
                    for(k=1:nt)
                        %--- listwise (P)
                        btmp = CAT(isfinite(sum(CAT,2)),k)' * (xall(isfinite(sum(CAT,2)),:) /( xall(isfinite(sum(CAT,2)),:)'*xall(isfinite(sum(CAT,2)),:) ));
                        bs_csect_P(bsr,k,1) = btmp(end);
                        %--- pairwise (P)
                        btmp = CAT(isfinite(CAT(:,k)),k)' * xall(isfinite(CAT(:,k)),:) /( xall(isfinite(CAT(:,k)),:)'*xall(isfinite(CAT(:,k)),:) );
                        bs_csect_P(bsr,k,2) = btmp(end);
                    end

                    %--- imputed (N,P)
                    vmat=[]; % concat time blocks
                    for(k=1:nt)
                        vmat = [vmat; [YT(listt,k), BLK(listt,:,k), [XT(listt,:) subix(listt)]]];
                    end
                    vmat=repmat(vmat,10,1); % whole thing x10
                    tab_new  = array2table( vmat, 'VariableNames',vnam2);
                    valimp = random( out.LMM.model, tab_new ); %imput data, then rearrange as nsubj x sample/replicate
                    valimp = reshape(valimp,numel(listt),[]); % (A,S,R,M,Y)#1, (A,S,R,M,Y)#2 ...
                    for(m=1:10)
                        valimp_sub      = valimp(:,(1:nt) + nt*(m-1)); % take each successive block of 5 subjxtimepoint
                        Ytmp            = YTbs; Ytmp(~isfinite(Ytmp))=0;
                        YTbs_imp        = Ytmp + valimp_sub .* double( ~isfinite(YTbs) );
                        %--- imputed (N)
                        xall = [[ones(sum(dxbs<-eps),1) CTbs(dxbs<-eps,:) ones(sum(dxbs<-eps),1)]; [ones(size(CCbs,1),1) CCbs zeros(size(CCbs,1),1)]];
                        CAT  = [YTbs_imp(dxbs<-eps,:); repmat(YCbs,1,nt)];
                        btmp = CAT' * xall /( xall'*xall );
                        estim_tmp_N(m,:) = btmp(:,end);
                        %--- imputed (P)
                        xall = [[ones(sum(dxbs> eps),1) CTbs(dxbs> eps,:) ones(sum(dxbs> eps),1)]; [ones(size(CCbs,1),1) CCbs zeros(size(CCbs,1),1)]];
                        CAT  = [YTbs_imp(dxbs> eps,:); repmat(YCbs,1,nt)];
                        btmp = CAT' * xall /( xall'*xall );
                        estim_tmp_P(m,:) = btmp(:,end);
                    end
                    bs_csect_N(bsr,:,3) = mean(estim_tmp_N,1); % averaged imputed estimate
                    bs_csect_P(bsr,:,3) = mean(estim_tmp_P,1); % averaged imputed estimate
                end

    % =================== if subgrouping *is* imposed on ctls ================= %      
            else

                dxcbs = dxc(listt);

                if(isempty(CT)) % -------------------------------------- unadjusted

                    %--- listwise (N,P)
                    bs_csect_N(bsr,:,1) = ( mean(YTbs(isfinite(sum(YTbs,2)) & dxbs<-eps,:),1) - mean(YCbs(dxcbs<-eps,:),1) );
                    bs_csect_P(bsr,:,1) = ( mean(YTbs(isfinite(sum(YTbs,2)) & dxbs> eps,:),1) - mean(YCbs(dxcbs> eps,:),1) );
                    %--- pairwise (N,P)
                    bs_csect_N(bsr,:,2) = ( mean(YTbs(dxbs<-eps,:),1,'omitnan') - mean(YCbs(dxcbs<-eps,:),1) );
                    bs_csect_P(bsr,:,2) = ( mean(YTbs(dxbs> eps,:),1,'omitnan') - mean(YCbs(dxcbs> eps,:),1) );
                    %--- imputed (N,P)
                    vmat=[]; % concat time blocks
                    for(k=1:nt)
                        vmat = [vmat; [YT(listt,k), BLK(listt,:,k), [XT(listt,:) subix(listt)]]];
                    end
                    vmat=repmat(vmat,10,1); % whole thing x10
                    tab_new  = array2table( vmat, 'VariableNames',vnam2);
                    valimp = random( out.LMM.model, tab_new ); %imput data, then rearrange as nsubj x sample/replicate
                    valimp = reshape(valimp,numel(listt),[]); % (A,S,R,M,Y)#1, (A,S,R,M,Y)#2 ...
                    for(m=1:10)
                        valimp_sub      = valimp(:,(1:nt) + nt*(m-1)); % take each successive block of 5 subjxtimepoint
                        Ytmp            = YTbs; Ytmp(~isfinite(Ytmp))=0;
                        YTbs_imp        = Ytmp + valimp_sub .* double( ~isfinite(YTbs) );
                        %
                        estim_tmp_N(m,:) = mean(YTbs_imp(dxbs<-eps,:),1) - mean(YCbs(dxcbs<-eps,:),1);
                        estim_tmp_P(m,:) = mean(YTbs_imp(dxbs> eps,:),1) - mean(YCbs(dxcbs> eps,:),1);
                    end
                    bs_csect_N(bsr,:,3) = mean(estim_tmp_N,1); % averaged imputed estimate
                    bs_csect_P(bsr,:,3) = mean(estim_tmp_P,1); % averaged imputed estimate

                else % ----------------------------------------- covariate adjusted

                    CTbs =CT( listt,:);
                    CCbs =CC( listc,:); 

                    xall = [[ones(sum(dxbs<-eps),1) CTbs(dxbs<-eps,:) ones(sum(dxbs<-eps),1)]; [ones(sum(dxcbs<-eps),1) CCbs(dxcbs<-eps,:) zeros(sum(dxcbs<-eps),1)]];
                    CAT  = [YTbs(dxbs<-eps,:); repmat(YCbs(dxcbs<-eps,:),1,nt)];
                    for(k=1:nt)
                        %--- listwise (N)
                        btmp = CAT(isfinite(sum(CAT,2)),k)' * (xall(isfinite(sum(CAT,2)),:) /( xall(isfinite(sum(CAT,2)),:)'*xall(isfinite(sum(CAT,2)),:) ));
                        bs_csect_N(bsr,k,1) = btmp(end);
                        %--- pairwise (N)
                        btmp = CAT(isfinite(CAT(:,k)),k)' * xall(isfinite(CAT(:,k)),:) /( xall(isfinite(CAT(:,k)),:)'*xall(isfinite(CAT(:,k)),:) );
                        bs_csect_N(bsr,k,2) = btmp(end);
                    end
                    %
                    xall = [[ones(sum(dxbs> eps),1) CTbs(dxbs> eps,:) ones(sum(dxbs> eps),1)]; [ones(sum(dxcbs> eps),1) CCbs(dxcbs> eps,:) zeros(sum(dxcbs> eps),1)]];
                    CAT  = [YTbs(dxbs> eps,:); repmat(YCbs(dxcbs> eps,:),1,nt)];
                    for(k=1:nt)
                        %--- listwise (P)
                        btmp = CAT(isfinite(sum(CAT,2)),k)' * (xall(isfinite(sum(CAT,2)),:) /( xall(isfinite(sum(CAT,2)),:)'*xall(isfinite(sum(CAT,2)),:) ));
                        bs_csect_P(bsr,k,1) = btmp(end);
                        %--- pairwise (P)
                        btmp = CAT(isfinite(CAT(:,k)),k)' * xall(isfinite(CAT(:,k)),:) /( xall(isfinite(CAT(:,k)),:)'*xall(isfinite(CAT(:,k)),:) );
                        bs_csect_P(bsr,k,2) = btmp(end);
                    end

                    %--- imputed (N,P)
                    vmat=[]; % concat time blocks
                    for(k=1:nt)
                        vmat = [vmat; [YT(listt,k), BLK(listt,:,k), [XT(listt,:) subix(listt)]]];
                    end
                    vmat=repmat(vmat,10,1); % whole thing x10
                    tab_new  = array2table( vmat, 'VariableNames',vnam2);
                    valimp = random( out.LMM.model, tab_new ); %imput data, then rearrange as nsubj x sample/replicate
                    valimp = reshape(valimp,numel(listt),[]); % (A,S,R,M,Y)#1, (A,S,R,M,Y)#2 ...
                    for(m=1:10)
                        valimp_sub      = valimp(:,(1:nt) + nt*(m-1)); % take each successive block of 5 subjxtimepoint
                        Ytmp            = YTbs; Ytmp(~isfinite(Ytmp))=0;
                        YTbs_imp        = Ytmp + valimp_sub .* double( ~isfinite(YTbs) );
                        %--- imputed (N)
                        xall = [[ones(sum(dxbs<-eps),1) CTbs(dxbs<-eps,:) ones(sum(dxbs<-eps),1)]; [ones(sum(dxcbs<-eps),1) CCbs(dxcbs<-eps,:) zeros(sum(dxcbs<-eps),1)]];
                        CAT  = [YTbs_imp(dxbs<-eps,:); repmat(YCbs(dxcbs<-eps,:),1,nt)];
                        btmp = CAT' * xall /( xall'*xall );
                        estim_tmp_N(m,:) = btmp(:,end);
                        %--- imputed (P)
                        xall = [[ones(sum(dxbs> eps),1) CTbs(dxbs> eps,:) ones(sum(dxbs> eps),1)]; [ones(sum(dxcbs> eps),1) CCbs(dxcbs> eps,:) zeros(sum(dxcbs> eps),1)]];
                        CAT  = [YTbs_imp(dxbs> eps,:); repmat(YCbs(dxcbs> eps,:),1,nt)];
                        btmp = CAT' * xall /( xall'*xall );
                        estim_tmp_P(m,:) = btmp(:,end);
                    end
                    bs_csect_N(bsr,:,3) = mean(estim_tmp_N,1); % averaged imputed estimate
                    bs_csect_P(bsr,:,3) = mean(estim_tmp_P,1); % averaged imputed estimate
                end
            end
        end
        % (N) now convert to summary stats
        for(w=1:3)
            difb = bs_csect_N(:,:,w);
            effix= isfinite(sum(difb,2));
            difb = difb(effix,:); size(difb),
            mattor = [mean(difb,1)', prctile(difb,[2.5 97.5])', (mean(difb)./std(difb))', (2*min( [mean(difb>0,1); mean(difb<0,1)],[],1 ))'];
            [~,th]=fdr( mattor(:,end),'p',0.05,0);
            out.xct_dif_N.(['stat_',misstype{w}]) = [mattor, th];
        end
        % (P) now convert to summary stats
        for(w=1:3)
            difb = bs_csect_P(:,:,w);
            effix= isfinite(sum(difb,2));
            difb = difb(effix,:); size(difb),
            mattor = [mean(difb,1)', prctile(difb,[2.5 97.5])', (mean(difb)./std(difb))', (2*min( [mean(difb>0,1); mean(difb<0,1)],[],1 ))'];
            [~,th]=fdr( mattor(:,end),'p',0.05,0);
            out.xct_dif_P.(['stat_',misstype{w}]) = [mattor, th];
        end
        disp('done!');
    
    end
    
end


end