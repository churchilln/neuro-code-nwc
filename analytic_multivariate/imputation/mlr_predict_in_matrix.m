function out = mlr_predict_in_matrix( X, lam )

[NS,NT] = size(X);

P = double(isfinite(X));
X(~isfinite(X))=0;

Ximpute = X; %% predefine data matrix to populate with imputed values

ix_missing_subj = find( sum(P==0,2)>0 );

for(i=1:numel(ix_missing_subj))
   
    Ptst = P(ix_missing_subj(i),:); % indexing presence/absence
    Xtst = X(ix_missing_subj(i),Ptst> 0); % collecting test datapoint

    ix_exist = find( mean(P(:,Ptst==0),2)==1 ); % all datapoints where missing variables are present
    Ptrn = P(ix_exist,:); % presence matrix for these datapoints
    
    % drop any Xtrn indexes / presence matrix entries where there none of the Xtst variables exist
    ix_exist( sum(Ptrn(:,Ptst> 0),2)==0    )=[];
    Ptrn(     sum(Ptrn(:,Ptst> 0),2)==0, : )=[];
    %  
    Ptrn = Ptrn(:,    Ptst> 0); % drop missing variable index - now presence values on remaining variables
    Xtrn = X(ix_exist,Ptst> 0); % variables present in test data are predictors
    Ytrn = X(ix_exist,Ptst==0); % response variables(s) are the missing ones in test point

    % group the types depending on # cases
    pindex = sum(bsxfun(@times, Ptrn, 2.^[1:size(Ptrn,2)]),2); % every combo gets a unique index
    pindex_unique = sort(unique(pindex),'descend');
    
    clear Ytst lev;
    
    for(n=1 : numel(pindex_unique) ) % fit a different PLS model to each case
        Ptrn_2   = Ptrn( pindex==pindex_unique(n), : ); 
        Ptrn_2   = Ptrn_2(1,:); % this is a unique combo, so only keep the first one in list
        Xtrn_2   = Xtrn( pindex==pindex_unique(n), Ptrn_2>0 ); % drop missing variables from this set
        Ytrn_2   = Ytrn( pindex==pindex_unique(n), : );
        
        if( size(Xtrn_2,1)>2 ) %% need at least three datapoints to make predictions
        
            % leverage exerted by model = #subjects x # variables
            lev(n,1) = (size(Xtrn_2,1)-1);

            % now fitting PLS model
            model = pmlr1( Xtrn_2, Ytrn_2, lam );
            % and making predictions
            pred  = pmlr2( Xtst, model );
            % store predicted values
            Ytst(n,:) = pred.Y_estim;
            
        else %% otherwise model gets zero leverage when averaging
            lev(n,1)  = 0;
            Ytst(n,:) = zeros(1,sum(Ptst==0));
        end      
    end
    
    lev  = lev./sum(lev); % normalized
    Ytst = lev'*Ytst;     % weighted mean
    % plug values into imputation matrix
    Ximpute( ix_missing_subj(i), Ptst==0 ) = Ytst;
end

out.Ximpute = Ximpute;

%%
function out = pmlr1( X, Y, lam )

% augmenting X
X = [ones(size(X,1),1), X];
% data dimensions
[S, V] = size(X);
[S, B] = size(Y);

for(b=1:B)        
    out.Beta(:,b) = (X'*X + eye(V)*lam ) \ (X'*Y(:,b));
end

%%
function out = pmlr2( X, model )

% augmenting X
X = [ones(size(X,1),1), X];
% store predicted values, in original format
out.Y_estim = X*model.Beta; 
