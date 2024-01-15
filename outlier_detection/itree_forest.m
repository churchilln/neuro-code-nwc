function tree = itree_forest( X,l, Niter, fullout, thval )
%
% isolation tree script
%
%    tree = itree( X,l,Niter,fullout,thval )
%
%    inputs:
%           X       : input data (samples x vars)
%           l       : maximum search level ... recommended min([8, nextpow2(#samples)])
%           fullout : generate all BSTs (otherwise only tree.ht, htav, scor)
%           thval   : thresholding value for determinining significant anomalies
%
%    outputs:
%
%           tree.mat{i} : binary clustering tree
%           tree.end{i} : terminal tree
%           tree.pix{i} : list of splitting variable indexes by level
%           tree.idx{i} : sample labels - each distinct branch has unique numeric label
%
%           tree.ht(:,i): tree height (search depth before sample enters singular branch)
%           tree.htav   : average tree height
%           tree.scor   : anomaly score
%

[N,P]=size(X);

for(iter= 1:Niter)

    terminflag=0;
    % first level - all in root
    step=0;
    treemat = ones(N,1);
    termmat = zeros(N,1);
    while(terminflag==0)

        step=step+1; % what level of tree

        idx = sum( bsxfun(@times,treemat,2.^fliplr(1:step)), 2); %unique index for all branches
        brnums = unique(idx); % number indices for distinct branches
        newbranch=zeros(N,1); % declare new (blank) branch
        newtermin=zeros(N,1); % terminal branch?

        p = ceil(rand(1)*P); % random feat.
        pmat(step,1) = p;
        
        clear numu;
        for(b=1:numel(brnums))
            xsub   = X( idx==brnums(b), p ); %% pull all subjects belonging t
            if(numel(unique(xsub)) > 1 )

                cut    = rand(1)*(range(xsub)-2*eps) + min(xsub)+eps;
                newbranch( idx==brnums(b) ) = double( xsub>=cut );
                numu(b,1) = numel(idx==brnums(b));
            else
                numu(b,1) = 1;
                newtermin( idx==brnums(b) ) = 1;
            end
        end
        treemat = [treemat newbranch];
        termmat = [termmat newtermin];

        if( step>=l || max(numu)==2 )
            terminflag=1;
        end
    end

    if( fullout>0 )
    tree.mat{iter} = treemat;
    tree.end{iter} = termmat;
    tree.pix{iter} = pmat;
    tree.idx{iter} = sum( bsxfun(@times,treemat,2.^fliplr(1:step+1)), 2);
    end
    % indexing at first split (ignores zeroth)
    tree.ht(:,iter)  = sum(bsxfun(@times, termmat(:,2:end) - termmat(:,1:end-1), 2:size(termmat,2)),2) - 1;
end

tree.ht(tree.ht<eps) = l;
% summary statistics
tree.htav = mean(tree.ht,2);
tree.scor = 2.^(-tree.htav./c(N));
tree.thr  = tree.scor >= thval; %% fixed threshold...

%% expected search length, random tree
function out = c( n )

out = ceil( 2*(log(n-1)+0.5772156649) - (2*(n-1)/n) );
