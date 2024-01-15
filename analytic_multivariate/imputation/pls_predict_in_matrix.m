function out = pls_predict_in_matrix( X, Kmax )

[NS,NT] = size(X);

P = double(isfinite(X));
X(~isfinite(X))=0;

Ximpute = X; %% predefine data matrix to populate with imputed values

ix_missing_subj = find( sum(P==0,2)>0 );

for(i=1:numel(ix_missing_subj))

    Ptst = P(ix_missing_subj(i),:);       % indexing presence/absence
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
            model = ppls1( Xtrn_2, Ytrn_2, Kmax, [1 1] );
            % and making predictions
            pred  = ppls2( Xtst, model );
            % store predicted values
            Ytst(n,:) = pred.Y_estim1;
            
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
function out = ppls1( X, Y, Kmax, standardize )

[S, V] = size(X);
[S, B] = size(Y);

Klim = min([S-1, V]);
if( isempty(Kmax) || Kmax>Klim )
    %disp('setting K...');
    Kmax = Klim;
end

deflate=[1 2];

if( B==1 && Kmax>1 && deflate(2)~=2 )
    error('for single behavioural variable, only "predictive mode" deflation allowed (deflate(2)=2)');
end

% input transformations
if(standardize(1)>=0)  out.Xav = mean(X,1);
else                   out.Xav = zeros(1,size(X,2));
end
if(standardize(1)>=1)  out.Xsd = std(X,0,1);
else                   out.Xsd = ones(1,size(X,2));
end
% target transformations
if(standardize(2)>=0)  out.Yav = mean(Y,1);
else                   out.Yav = zeros(1,size(Y,2));
end
if(standardize(2)>=1)  out.Ysd = std(Y,0,1);
else                   out.Ysd = ones(1,size(Y,2));
end
% apply transformations
X = bsxfun(@rdivide,bsxfun(@minus,X,out.Xav),out.Xsd); 
Y = bsxfun(@rdivide,bsxfun(@minus,Y,out.Yav),out.Ysd); 

Xup = X; %-duplicate for deflation
Yup = Y; %-duplicate for deflation

Wx  = randn(V,Kmax); %-weighs(X-vox)
Wy  = randn(B,Kmax); %-weighs(Y)
Cx  = randn(S,Kmax); %-scores(X)
Cy  = randn(S,Kmax); %-scores(Y)
Px  = randn(V,Kmax); %-loadings(X-vox)
Py  = randn(B,Kmax); %-loadings(Y)
By  = randn(B,Kmax); %-loadings(Y) -- for Y-predictive mode

%init fixed??
Wy(:,1) = 1/B;
Cy(:,1) = Yup*Wy(:,1);

nonconv=0; %% nonconvergence checker (only used for Kmax>1)

yvar_init = sum(Yup(:).^2);


for(k=1:Kmax)

    % estimate weighting vectors
    [u,~,v]= svd( Xup'*Yup, 'econ' );
    Wx(:,k) = u(:,1);
    Wy(:,k) = v(:,1);
    % score vectors
    Cx(:,k) = Xup*Wx(:,k);
    Cy(:,k) = Yup*Wy(:,k);
    % regression fits
    Px(:,k) = (Xup'*Cx(:,k)) ./ (Cx(:,k)'*Cx(:,k));
    Py(:,k) = (Yup'*Cy(:,k)) ./ (Cy(:,k)'*Cy(:,k));

    % deflating on X
    Xup = Xup - Cx(:,k)*Px(:,k)';
    % deflating on Y...
    Py(:,k) = (Yup'*Cy(:,k)) ./ (Cy(:,k)'*Cy(:,k));
    By(:,k) = (Yup'*Cx(:,k)) ./ (Cx(:,k)'*Cx(:,k));
    if    ( deflate(2)==1 ) %% symmetric mode
        Yup = Yup - Cy(:,k)*Py(:,k)';
    elseif( deflate(2)==2 ) %% predictive mode
        Yup = Yup - Cx(:,k)*By(:,k)';
    end

    yvar_rem(k,1) = sum(Yup(:).^2);  
end

% store variables to outputs
out.Wx = Wx;
out.Wy = Wy;
out.Cx = Cx;
out.Cy = Cy;
out.Px = Px;
out.Py = Py;
out.By = By;
% now get full-model regression coefficients
out.Wt_predict = sum( Cx.*Cy ) ./ sum( Cx.^2 );
out.Wt_bpls    = (Y'*Cx) / (Cx'*Cx);
% and variable importance in the projections
out.yvarfrac = (yvar_rem - [yvar_init; yvar_rem(1:end-1)])./yvar_init; % frac explan

%%
function out = ppls2( X, model )

[S, V] = size(X);

Kmax = size(model.Cx,2);

deflate = [1 2];

% apply transformations
X = bsxfun(@rdivide,bsxfun(@minus,X,model.Xav),model.Xsd); 

Xup = X; %-duplicate for deflation

for(k=1:Kmax)
    % predicted subject scores
    Cx_pred(:,k) = Xup*model.Wx(:,k);
    % deflating on x
    if(deflate(1)==1)
        Xup = Xup - Cx_pred(:,k)*model.Px(:,k)';
    end   
end

Y_estim1=0;
Y_estim2=0;
% now make-a the predictions
for(k=1:Kmax)
    Y_estim1 = Y_estim1 + model.Wt_predict(k) * Cx_pred(:,k)*model.Py(:,k)';
    Y_estim2 = Y_estim2 + Cx_pred(:,k) * model.Wt_bpls(:,k)';
end

% store predicted values, in original format
out.Y_estim1 = bsxfun(@times,bsxfun(@plus,Y_estim1,model.Yav),model.Ysd); 
out.Y_estim2 = bsxfun(@times,bsxfun(@plus,Y_estim2,model.Yav),model.Ysd); 
