function out = NPLSC( Xcell, Ycell, N_iters, var_norm, NF, fdr_q )
% .

error('this file is currently a work in progress! check back later!')

%% initialization

% convert to cells if matrix-struct
if(~iscell(Xcell)) Xcell = {Xcell}; end
if(~iscell(Ycell)) Ycell = {Ycell}; end

% data checking
if(numel(Xcell)~=numel(Ycell))
    error('number of groups (cells) does not match for X and Y');
end
Ng = numel(Xcell);
for(g=1:Ng)
    dtmp = size(Xcell{g});
     Nx(g,1) = dtmp(1);
     if( g>1 && numel(dtmp(2:end)) ~= size(Pxset,2)) 
         error('input tensors X must by of same order');
     end
     Px(g,:) = dtmp(2:end);
    [Ny(g,1) Py(g,1)] = size(Ycell{g});
end
Nord = size(Pxset,2);

if(numel(unique([Nx(:); Ny(:)]))>1)
    error('sample sizes (number of rows) must be consistent for all X and Y matrices');
    disp('Samples in X:');
    disp(Nx');
    disp('Samples in Y:');
    disp(Ny');
end
Ns = Nx(1); clear Nx Ny;
for(o=1:Nord)
    if(numel(unique(Px(:,o)))>1)
        error('Number of variables (number of cols) must be consistent for all X matrices');
        disp(['Variables in X, ord=',num2str(o),':']);
        disp(Px(:,o)');
    end
end
Px = Px(1,:);
    
Pm    = min([sum(Px) sum(Py)]); % maximal rank is not well-defined here, but this is a conservative bound
match = 0; % flag to choose which variable set to match on (0=x,1=y)

% checking for options / if none, use default settings
if( nargin<3 ) N_iters  = 1000; disp('default: 1000 resampling iterations'); end
if( nargin<4 ) var_norm =    2; disp('default: normalizing variance'); end
if( nargin<5 ) NF = Pm;
               disp(['default: model max. number of latent factors NF=',num2str(Pm)]);
else
               if(NF>Pm) disp('too many factors requested, adjusting...'); end
               NF = min([NF Pm]);
               disp(['number of latent factors NF=',num2str(NF)]);
end
if( nargin<6 ) fdr_q=0.05; disp('default: FDR=.05 threshold'); end

if(match==0) ref='X'; elseif(match==1) ref='Y'; end
disp(['matching components based on ',ref,' data matrix']);

% simple Imputation - replace missing values with median
if( sum(isnan(Xcell{g}(:)))>0 )
    disp(['Found ',num2str(sum(isnan(Xcell{g}(:)))),' missing entries in X. Imputing...']);
    Xcell{g} = reshape( Xcell{g}, Ns, [] );
    for(p=1:size(Xcell{g},2))
       tmp=Xcell{g}(:,p); 
       tmp(~isfinite(tmp))=median(tmp(isfinite(tmp))); 
       Xcell{g}(:,p)=tmp; 
    end
    Xcell{g} = reshape( Xcell{g}, [Ns Px] );
    %X = knnimpute(X')'; 
end
if( sum(isnan(Ycell{g}(:)))>0 )
    disp(['Found ',num2str(sum(isnan(Ycell{g}(:)))),' missing entries in Y. Imputing...']);    
    for(p=1:Py)
       tmp=Ycell{g}(:,p); 
       tmp(~isfinite(tmp))=median(tmp(isfinite(tmp))); 
       Ycell{g}(:,p)=tmp; 
    end
    %Y = knnimpute(Y')'; 
end

% if rank-normalization specified, apply to data
if( var_norm == 3 )
    for(g=1:Ng)
        Xcell{g} = reshape( tiedrank( reshape( Xcell{g}, Ns, [] ) ), [Ns Px] );
        Ycell{g} = tiedrank(Ycell{g});
    end
end    

% whole-data analysis for reference    
[uo,lo,lfo,vo] = svd_tensprod( Xcell, Ycell, NF )

%% permutation analysis

if(sum(Py)>1)

    disp('running permutation testing...');

    for(iter=1:N_iters)

       disp(['resample ', num2str(i),' of ', num2str(N_iters)]);
       listx  = randperm(Nx);
       listy  = randperm(Ny);

       for(g=1:Ng) 
           Xprm{g} = Xcell{g}(listx,:); 
           Yprm{g} = Ycell{g}(listy,:); 
       end
       [~,l_prm(:,iter),lf_prm(:,iter),~] = svd_xprod( Xprm, Yprm, NF );
    end
    
    out.p_l  = mean( bsxfun(@gt,l_prm,lo), 2 );
    out.p_lf = mean( bsxfun(@gt,lf_prm,lfo), 2 );
else
    disp('cannot do permutation analysis'); 
end

%% bootstrap analysis

% predeclare matrices
ub_set = zeros(Px,NF,N_iters);
vb_set = zeros(sum(Py),NF,N_iters);
sb_set = zeros(NF,N_iters);

disp('running bootstrap testing...');
for(iter=1:N_iters)
    
    disp(['resample ', num2str(i),' of ', num2str(N_iters)]);
    list  = ceil(Ns*rand(Ns,1));
    for(g=1:Ng) 
        Xbs{g} = Xcell{g}(list,:); 
        Ybs{g} = Ycell{g}(list,:); 
    end
    [u,l,lf,v] = svd_xprod( Xbs, Ybs, NF );

    % matching latent variable sets
    if    (match==0) %match on u (saliences of X)            
        ob=mini_procrust_ex(uo,u,'corr');
    elseif(match==1) %match on v (saliences of Y)
        ob=mini_procrust_ex(vo,v,'rss');
    end
    % store (rearranged) component sets
    ub_set(:,:,iter)=u(:,ob.index)*diag(ob.flip);
    vb_set(:,:,iter)=v(:,ob.index)*diag(ob.flip);
    sb_set(:,iter)  =l(ob.index);
    fb_set(:,iter)  =lf(ob.index);
end

out.u_bsr  = mean(ub_set,3)./(std(ub_set,0,3)+eps);
[p out.u_fdr] = fdr_ex( out.u_bsr, 'z', fdr_q, 0 );

if(sum(Py)>1)
    out.v_bsr     = mean(vb_set,3)./std(vb_set,0,3);
    [p out.v_fdr] = fdr_ex( out.v_bsr, 'z', fdr_q, 0 );
    % var fraction
    out.sb_avg    = mean(sb_set,2);
    out.sb_95ci   = prctile(sb_set,[2.5 97.5],2);
    out.fb_avg    = mean(fb_set,2);
    out.fb_95ci   = prctile(fb_set,[2.5 97.5],2);
end

function [u,l,lf,v] = svd_tensprod( Xcell, Ycell, NF )

dtmp = size(Xcell{1});
ns=dtmp(1);
px=dtmp(2:end);
nd=numel(px);
py=size(Ycell{1},2);

% setting up the full product tensor
xprod = [];
for(g=1:numel(Xcell))
    % standardized
    xtmp = bsxfun(@minus,Xcell{g},mean(Xcell{g},1));
    xtmp = bsxfun(@rdivide,xtmp,sqrt(sum(xtmp.^2,1)));
    ytmp = bsxfun(@minus,Ycell{g},mean(Ycell{g},1));
    ytmp = bsxfun(@rdivide,ytmp,sqrt(sum(ytmp.^2,1)));
    % linear products
    tmpprod=[];
    for(h=1:py)
       tmpprod = cat(1, sum( bsxfun(@times,xtmp,ytmp(:,h)), 1));
    end
    
    % concatenate along behavioural dimension
    xprod = cat(nd+1,xprod, tmpprod);
end
% product tensor ( dy(1) x dx(1) x dx(2) x ... dx(nd) x dg )

if(size(xprod,1)==1)    uvlab(1) = 1;
else                    uvlab(1)=0;
end
if(size(xprod,nd+1)==1) uvlab(2) = 1;
else                    uvlab(2)=0;
end

xprod = squeeze( xprod ); % reduced tensor -- assumes no x to squeeze
dset  = size(xprod); % size of each dim
ndnu  = numel(dset); % number of dims (order)

% initialized
for(i=1:ndnu)
   uset{i} = randn(dset(i),NF); 
   uset{i} = bsxfun(@rdivide,uset{i},sqrt(sum(uset{i}.^2)));
end

iter=0;
conv=1;

while( iter<200 && conv>1E-6 )

    % go through fitting cycle
    for(i=1:ndnu)
        utmp = uset( circshift(1:ndnu,-1) );
        xtmp = shiftdim(xprod,1);
        for(j=2:ndnu)
           xtmp = sum( bsxfun(@times,xtmp, permute( utmp{j}(:,1), [2:j, 1] ) ), j );
        end
        uset{i}(:,1) = xtmp ./ norm(xtmp);
    end
    
    
end

% [u,l,v]=svd( xprod,'econ' );
% u=u(:,1:NF);
% l=diag(l(1:NF,1:NF));
% lf=diag(l(1:NF,1:NF))./trace(l);
% v=v(:,1:NF);

%%
function [ Out ] = mini_procrust_ex( refVects, subVects, type )
%
% VERY simple version of procrustes - matches subVects to most appropriate
% refVect, in order to minimize global SumSquares difference criteria
%

% get dimensions from subspace-vectors
nVct     = size( subVects,2);
subV_idx = zeros(nVct,1);

if( strcmp( type , 'rss' ) )

    % get reference vector ordering, by decreasing variance
    ordRef  = sortrows( [ (1:nVct)' std( refVects )'], -2 );
    ordRef  = ordRef(:,1);

    % step through the reference vectors
    for( ir=1:nVct )
        % replicate out the reference
        tmpRef   = repmat( refVects(:,ir), [1 nVct] );
        % get the sum-of-squares difference from each reference vector (flipping to match by sign)
        SS(ir,:) = min( [sum((subVects - tmpRef).^2)', sum((-subVects - tmpRef).^2)'], [], 2 );
    end

    % we have sum-of-sqr difference matrix SS = ( nref x nsub )
    
    % step through reference vectors again (now by amt of explained var.)
    for(j=1:nVct)    
        % find the sub-vector index minimizing deviation from Ref
        [vs is] = min( SS(ordRef(j),:) );
        % for this Ref-vector, get index of best match SubVect
        subV_idx( ordRef(j) ) = is;
        % then "blank out" this option for all subsequent RefVects
        SS( :, is ) = max(SS(SS~=0)) + 1;
    end

    % reordered to match their appropriate RefVects
    subVects_reord = subVects(:,subV_idx);
    % now recontstruct what the sign was via index
    [vvv iii] = min([sum((-subVects_reord - refVects).^2)', sum((subVects_reord - refVects).^2)'], [], 2);
    % convert to actual sign
    flip= sign(iii-1.5);
    % output:
    % 
    Out.index  = subV_idx(:);
    Out.flip   = flip(:);

elseif( strcmp( type , 'corr' ) )
    
    ordRef  = (1:nVct)';

    % full correlations [ref x sub]
    CC = abs(corrcoef( [refVects, subVects] ));    
    CC = CC(1:nVct, nVct+1:end);
    
    remainRef = (1:nVct)'; ordRef = [];
    remainSub = (1:nVct)'; ordSub = [];
    
    CCtmp = CC;
    
    for( i=1:nVct )
        
        % get max correlation of ea. ref
        [vMax iMax] = max( CCtmp,[], 2 );
        % find Ref with highest match
        [vOpt iRef] = max( vMax     );
        % also get "sub" index:
              iSub  = iMax(iRef);
        
        ordRef = [ordRef remainRef( iRef )];
        remainRef( iRef ) = [];
        %
        ordSub = [ordSub remainSub( iSub )];
        remainSub( iSub ) = [];
        %
        CCtmp(iRef,:) = [];
        CCtmp(:,iSub) = [];
    end
    
    CCnew = corrcoef( [refVects(:,ordRef) subVects(:,ordSub)] );
    flip  = sign( diag( CCnew(1:nVct,nVct+1:end) ) );
    
    resort = sortrows([ordRef(:) ordSub(:) flip(:)], 1);

    Out.index = resort(:,2);
    Out.flip   = resort(:,3);    
end

%%
function [pcritSet threshMat] = fdr_ex( dataMat, datType, qval, cv, dof )
% 
% False-Discovery Rate correction for multiple tests
% for matrix of some test statistic, get back binary matrix where
% 1=signif. at FDR thresh / 0=nonsignif
% 
%   [pcrit threshMat] = fdr( dataMat, datType, qval, cv, dof )
%
%   where datType -> 'p': p-value
%                    't': t-score -- requires a dof value
%                    'z': z-score
%
%         qval    -> level of FDR control (expected fdr rate)
% 
%         cv      -> constant of 0:[cv=1]  1:[cv= sum(1/i)]
%                    (1) applies under any joint distribution of pval
%                    (0) requires relative test indep & Gaussian noise with
%                        nonnegative correlation across tests
%         dof     -> degrees of freedom, use to assess significance of t-stat
%
%   * get back testing matrix threshMat & critical pvalue pcrit
%   * testing is currently 2-tail only!
%   * dof value is only relevant if you are using tstats
%

% ------------------------------------------------------------------------%
% Authors: Nathan Churchill, University of Toronto
%          email: nathan.churchill@rotman.baycrest.on.ca
%          Babak Afshin-Pour, Rotman reseach institute
%          email: bafshinpour@research.baycrest.org
% ------------------------------------------------------------------------%
% CODE_VERSION = '$Revision: 158 $';
% CODE_DATE    = '$Date: 2014-12-02 18:11:11 -0500 (Tue, 02 Dec 2014) $';
% ------------------------------------------------------------------------%

[Ntest Nk] = size(dataMat);

if    ( datType == 'p' )  probMat = dataMat;
elseif( datType == 't' )  probMat = 1-tcdf( abs(dataMat), dof );
elseif( datType == 'z' )  probMat = 1-normcdf( abs(dataMat) );    
end

threshMat = zeros( Ntest, Nk );

for( K=1:Nk )

    % (1) order pvalues smallest->largest
    pvect = sort( probMat(isfinite( probMat(:,K) ), K), 'ascend' );
    Ntest2= length(pvect);
    % (2) find highest index meeting limit criterion
    if(cv == 0) c_V = 1;
    else        c_V = log(Ntest2) + 0.5772 + 1/Ntest2; % approx soln.
    end
    % index vector
    indxd = (1:Ntest2)';
    % get highest index under adaptive threshold
    r = sum( pvect./indxd <= qval/(Ntest2*c_V) );

    if( r > 0 )
        % limiting p-value
        pcrit = pvect(r);
        % threshold matrix values based on prob.        
        threshMat(:,K) = double(probMat(:,K)  <= pcrit);
        % critical p-values        
        pcritSet(K,1)  = pcrit;
    else
        threshMat(:,K) = zeros(Ntest,1);
        pcritSet(K,1)  = NaN;
    end
    
end
