function out = matfac_als( X, K, initStruct, bias_terms, lambda, Ntry, orth  )
%
% matrix factorization for imputation of 2-way and 3-way datasets
% assumed structure: [V x S] or [V x T x S]
% ...this is relevant because scaling gets put into "S" domain!
%

if( nargin<3 ) initStruct = []; end
if( nargin<4 ) bias_terms = [0 0 0]; end
if( nargin<5 ) lambda     = [1E-9 1E-9 1E-9]; end
if( nargin<6 ) Ntry       = 5; end
if( nargin<7 ) orth       = 0; end

if(size(X,3)==1) 
    bias_terms=bias_terms(1:2);
    lambda=lambda(1:2);
end

for(n=1:Ntry)
    
   disp([' Try #',num2str(n),' : ']);
   
   if(K==0) out_tmp = run_matfac_biasonly( X,  initStruct                         );
   else     out_tmp =          run_matfac( X,K,initStruct,bias_terms,lambda, orth );
   end
   
   if( n==1 ) 
       out = out_tmp; 
   elseif( out_tmp.mod_err(end) < out.mod_err ) 
       out = out_tmp;
   end
end

%%
function out = run_matfac( X, K, initStruct, bias_terms, lambda, orth )

if(size(X,3)==1)

    if( isempty(lambda) ) 
        lv=1E-9;
        ls=1E-9;
    elseif( numel(lambda)~=2)
        error('for matrix input, need 2 regularization terms (or none)');
    else
        lv=lambda(1);
        ls=lambda(2);
    end
    
    [NV,NS] = size(X);

    P = double(isfinite(X));
    X(~isfinite(X))=0;

    Vk = randn(NV,K);
    Sk = randn(NS,K);
    bv = zeros(NV,1);
    bs = zeros(NS,1);

    %%
    
    dev_modl=1E9;
    dev_imput=1E9;

        % start by imputed row/col means
        Xest  = bsxfun(@rdivide,sum(X.*P,1),sum(P,1)+eps) + bsxfun(@rdivide,sum(X.*P,2),sum(P,2)+eps) - sum(sum(X.*P))./(sum(sum(P))+eps);

        if    ( strcmpi( initStruct.type, 'nonoise' ) )
            % do nothing...
        elseif( strcmpi( initStruct.type, 'randn' ) ) 
            % inject random white noise
            mu   = mean(X(P>0));
            sd   =  std(X(P>0));
            Xest = Xest + (randn(size(Xest))+mu) .* (sd * initStruct.scal);
        else
            error('???');
        end
 
        Xest_old = Xest;
        % update matrix with imputed values
        X = X.*P + Xest.*(1-P);
        
        iter_out=0;

       while(iter_out<10 && dev_imput > 1E-6 )

% % %            %%% renormalization...
% % %            X = bsxfun(@rdivide,X,sqrt(sum(sum(X.^2,1),2)));
% % %            %%% renormalization...
           
           iter_out=iter_out+1;
           iter=0;
           dev_modl=1E9;
           clear mod_err;

           while(iter<100 && dev_modl > 1E-6 ) %% all model fitting now done on the imputed matrix -- drop the P weighting
                iter=iter+1;

                resid = X - bsxfun(@plus,bsxfun(@plus, Vk*Sk', bv), bs');
                mod_err(iter,1) = sum(sum(resid.^2)) + lv*trace(Vk'*Vk) + ls*trace(Sk'*Sk);

                if(bias_terms(1)>0)
                bv = sum(bsxfun(@minus,X-(Vk*Sk'),bs'),2) ./ NS; bv=bv(:);
                end
                if(bias_terms(2)>0)
                bs = sum(bsxfun(@minus,X-(Vk*Sk'),bv ),1) ./ NV; bs=bs(:);
                end
                
                Vk = ((X  - bv*ones(NS,1)' - ones(NV,1)*bs' )*Sk) / ( Sk'*Sk + lv*eye(K) );
                Sk = ((X' - bs*ones(NV,1)' - ones(NS,1)*bv' )*Vk) / ( Vk'*Vk + ls*eye(K) );

                if(iter>1)
                   dev_modl = abs( (mod_err(iter)-mod_err(iter-1))./mod_err(iter-1) );
                end
           end

           % now refine the imputations
           Xest = bsxfun(@plus,bsxfun(@plus, Vk*Sk', bv), bs');
           % update matrix with imputed values
           X = X.*P + Xest.*(1-P);

           dev_imput = sum(sum( (Xest(P==0) - Xest_old(P==0)).^2 ))/sum(P(:)==0);
           Xest_old = Xest;
       end

    %disp(['ALS terminated. iter# ',num2str(iter),' -- emIter#',num2str(iter_out),' -- dev: ',num2str(dev_modl),' -- err: ',num2str(mod_err(end))])

    %--- rescaling
    vscal = sqrt(sum(Vk.^2));
    Vk = bsxfun(@rdivide,Vk,vscal);
    Sk = bsxfun(@times,Sk,vscal);

    out.mod_err = mod_err;
    out.Vk      = Vk;
    out.Sk      = Sk;
    out.bs      = bs;
    out.bv      = bv;
    out.reco    = bsxfun(@plus,bsxfun(@plus, Vk*Sk', bv), bs');
    %
    out.impute   = X.*P + out.reco.*(1-P);
    out.merr_av  = sum(sum(sum( (X-out.reco).^2 .*   P  ,1), 2),3)./sum(  P(:)  );
else

    if( isempty(lambda) ) 
        lv=1E-9;
        lt=1E-9;
        ls=1E-9;
    elseif( numel(lambda)~=3)
        error('for matrix input, need 2 regularization terms (or none)');
    else
        lv=lambda(1);
        lt=lambda(2);
        ls=lambda(3);
    end

    [NV,NT,NS] = size(X);

    P = double(isfinite(X));
    X(~isfinite(X))=0;

    Vk = randn(NV,K);
    Tk = randn(NT,K);
    Sk = randn(NS,K);
    %
    bv = zeros(NV,1);
    bt = zeros(NT,1);
    bs = zeros(NS,1);

    
    dev_modl=1E9;
    dev_imput=1E9;

        % start by imputed row/col means
        Xest  = bsxfun(@plus, bsxfun(@plus, bsxfun(@rdivide,sum(X.*P,1),sum(P,1)+eps), bsxfun(@rdivide,sum(X.*P,2),sum(P,2)+eps) ), bsxfun(@rdivide,sum(X.*P,3),sum(P,3)+eps) ) - sum(sum(sum(X.*P)))./(sum(sum(sum(P)))+eps );
        Xest_old = Xest;
        % update matrix with imputed values
        X = X.*P + Xest.*(1-P);

        iter_out=0;

       while(iter_out<10 && dev_imput > 1E-6 )

% % %            %%% renormalization...
% % %            X = bsxfun(@rdivide,X,sqrt(sum(sum(X.^2,1),2)));
% % %            %%% renormalization...           
           
           iter_out=iter_out+1;
           iter=0;
           dev_modl=1E9;
           clear mod_err;

           % initial estimate
           compfit=0; for(k=1:K) compfit = compfit+bsxfun(@times,Vk(:,k)*Tk(:,k)', permute(Sk(:,k),[3 2 1])); end

           while(iter<200 && dev_modl > 1E-6 ) %% all model fitting now done on the imputed matrix -- drop the P weighting
                iter=iter+1;

                resid = bsxfun(@minus,bsxfun(@minus,bsxfun(@minus,X,bv),bt'), permute(bs,[3 2 1])) - compfit;
                mod_err(iter,1) = sum(sum(sum(resid.^2))) + lv*trace(Vk'*Vk) + lt*trace(Tk'*Tk) + ls*trace(Sk'*Sk);

                
% % %                 xsub = bsxfun(@minus,bsxfun(@minus,bsxfun(@minus,X,bv),bt'),permute(bs,[3 2 1]));
% % %                 [Vk,Tk,Sk,~,~] = parafac(reshape(xsub,NV,[]),[NV NT NS],K,1E-6,[2 1 0],Vk,Tk,Sk);
                % vk
                xsub = bsxfun(@minus,bsxfun(@minus,bsxfun(@minus,X,bv),bt'),permute(bs,[3 2 1]));
                tmp1 = 0; for(s=1:NS) tmp1=tmp1+ xsub(:,:,s)*bsxfun(@times, Tk, Sk(s,:) ); end
                tmp2 = 0; for(s=1:NS) tmp2=tmp2+ (diag( Sk(s,:) )*Tk')*(Tk*diag( Sk(s,:) )); end
                if(orth==0)
                Vk = tmp1 * pinv(tmp2 + lv*eye(K) );
                elseif(orth>0)
                Vk = tmp1 * ((tmp1'*tmp1) + lv*eye(K) )^(-0.5);
                end
                % tk
                xsub = permute(xsub,[2 1 3]); % T x V x S
                tmp1 = 0; for(s=1:NS) tmp1=tmp1+ xsub(:,:,s)*bsxfun(@times, Vk, Sk(s,:) ); end
                tmp2 = 0; for(s=1:NS) tmp2=tmp2+ (diag( Sk(s,:) )*Vk')*(Vk*diag( Sk(s,:) )); end
                Tk = tmp1 / (tmp2 + lt*eye(K) );
                % sk
                xsub = permute(xsub,[3 2 1]); % S x V x T
                tmp1 = 0; for(s=1:NT) tmp1=tmp1+ xsub(:,:,s)*bsxfun(@times, Vk, Tk(s,:) ); end
                tmp2 = 0; for(s=1:NT) tmp2=tmp2+ (diag( Tk(s,:) )*Vk')*(Vk*diag( Tk(s,:) )); end
                Sk = tmp1 / (tmp2 + ls*eye(K) );

                % only update before bias estimation
                compfit=0; for(k=1:K) compfit = compfit+bsxfun(@times,Vk(:,k)*Tk(:,k)', permute(Sk(:,k),[3 2 1])); end
                
                % bv
                if(bias_terms(1)>0)
                resid = bsxfun(@minus,bsxfun(@minus,X,bt'), permute(bs,[3 2 1])) - compfit;
                bv = sum(sum(resid,2),3)./(NT*NS);
                end
                % bt
                if(bias_terms(2)>0)
                resid = bsxfun(@minus,bsxfun(@minus,X,bv), permute(bs,[3 2 1])) - compfit;
                bt = sum(sum(resid,1),3)./(NV*NS); bt=bt(:);
                end
                % bs
                if(bias_terms(3)>0)
                resid = bsxfun(@minus,bsxfun(@minus,X,bv),bt') - compfit;
                bs = sum(sum(resid,1),2)./(NV*NT); bs=bs(:);
                end

                if(iter>1)
                   dev_modl = abs( (mod_err(iter)-mod_err(iter-1))./mod_err(iter-1) );
                end
           end

           % now refine the imputations
           Xest = bsxfun(@plus,bsxfun(@plus,bv,bt'), permute(bs,[3 2 1]));
           for(k=1:K) Xest = Xest + bsxfun(@times,Vk(:,k)*Tk(:,k)', permute(Sk(:,k),[3 2 1])); end
           % update matrix with imputed values
           X = X.*P + Xest.*(1-P);

           dev_imput = sum(sum( (Xest(P==0) - Xest_old(P==0)).^2 ))/sum(P(:)==0);
           Xest_old = Xest;
       end

    %disp(['ALS terminated. iter# ',num2str(iter),' -- emIter#',num2str(iter_out),' -- dev: ',num2str(dev_modl),' -- err: ',num2str(mod_err(end))])

    %--- rescaling
    vscal = sqrt(sum(Vk.^2));
    tscal = sqrt(sum(Tk.^2));
    Vk = bsxfun(@rdivide,Vk,vscal);
    Tk = bsxfun(@rdivide,Tk,tscal);
    Sk = bsxfun(@times,Sk,tscal.*vscal);
    
    out.mod_err = mod_err;
    out.Vk = Vk;
    out.Tk = Tk;
    out.Sk = Sk;
    out.bv = bv;
    out.bt = bt;
    out.bs = bs;

    reco = bsxfun(@plus,bsxfun(@plus,bv,bt'), permute(bs,[3 2 1]));
    for(k=1:K) reco = reco + bsxfun(@times,Vk(:,k)*Tk(:,k)', permute(Sk(:,k),[3 2 1])); end
    out.reco = reco;
    %
    out.impute  = X.*P + out.reco.*(1-P);
    out.merr_av  = sum(sum(sum( (X-out.reco).^2 .*   P  ,1), 2),3)./sum(  P(:)  );
end

%     out.merr_s = permute( sum(sum( (X-out.reco).^2 .* P,1), 2)./sum(sum(P,1),2), [3 2 1]);
%     out.lev_s  = sum( (out.Sk*pinv(out.Sk'*out.Sk)).*out.Sk,2);

%%
function out = run_matfac_biasonly( X, initStruct )

if(size(X,3)==1)
    
    [NV,NS] = size(X);

    P = double(isfinite(X));
    X(~isfinite(X))=0;
    
    bv = randn(NV,1);
    bs = randn(NS,1);

    %%
    
    dev_modl=1E9;
    dev_imput=1E9;

        % start by imputed row/col means
        Xest  = bsxfun(@rdivide,sum(X.*P,1),sum(P,1)+eps) + bsxfun(@rdivide,sum(X.*P,2),sum(P,2)+eps) - sum(sum(X.*P))./(sum(sum(P))+eps);
        Xest_old = Xest;
        % update matrix with imputed values
        X = X.*P + Xest.*(1-P);

        iter_out=0;

       while(iter_out<10 && dev_imput > 1E-6 )

           iter_out=iter_out+1;
           iter=0;
           dev_modl=1E9;

           while(iter<100 && dev_modl > 1E-6 ) %% all model fitting now done on the imputed matrix -- drop the P weighting
                iter=iter+1;

                resid = X - bsxfun(@plus,bv, bs');
                mod_err(iter,1) = sum(sum(resid.^2));

                bv = sum(bsxfun(@minus,X,bs'),2) ./ NS; bv=bv(:);
                bs = sum(bsxfun(@minus,X,bv ),1) ./ NV; bs=bs(:);
                
                if(iter>1)
                   dev_modl = abs( (mod_err(iter)-mod_err(iter-1))./mod_err(iter-1) );
                end
           end

           % now refine the imputations
           Xest = bsxfun(@plus,bv, bs');
           % update matrix with imputed values
           X = X.*P + Xest.*(1-P);

           dev_imput = sum(sum( (Xest(P==0) - Xest_old(P==0)).^2 ))/sum(P(:)==0);
           Xest_old = Xest;
       end

    %disp(['ALS terminated. iter# ',num2str(iter),' -- emIter#',num2str(iter_out),' -- dev: ',num2str(dev_modl),' -- err: ',num2str(mod_err(end))])

    out.mod_err = mod_err;
    out.bs      = bs;
    out.bv      = bv;
    out.reco    = bsxfun(@plus,bsxfun(@plus, zeros(size(X)), bv), bs');
    %
    out.impute  = X.*P + out.reco.*(1-P);
    out.merr_av  = sum(sum(sum( (X-out.reco).^2 .*   P  ,1), 2),3)./sum(  P(:)  );
else

    [NV,NT,NS] = size(X);

    P = double(isfinite(X));
    X(~isfinite(X))=0;

    bv = randn(NV,1);
    bt = randn(NT,1);
    bs = randn(NS,1);

    
    dev_modl=1E9;
    dev_imput=1E9;

        % start by imputed row/col means
        Xest  = bsxfun(@plus, bsxfun(@plus, bsxfun(@rdivide,sum(X.*P,1),sum(P,1)+eps), bsxfun(@rdivide,sum(X.*P,2),sum(P,2)+eps) ), bsxfun(@rdivide,sum(X.*P,3),sum(P,3)+eps) ) - sum(sum(sum(X.*P)))./(sum(sum(sum(P)))+eps );
        Xest_old = Xest;
        % update matrix with imputed values
        X = X.*P + Xest.*(1-P);

        iter_out=0;

       while(iter_out<10 && dev_imput > 1E-6 )

           iter_out=iter_out+1;
           iter=0;
           dev_modl=1E9;

           while(iter<200 && dev_modl > 1E-6 ) %% all model fitting now done on the imputed matrix -- drop the P weighting
                iter=iter+1;

                resid = bsxfun(@minus,bsxfun(@minus,bsxfun(@minus,X,bv),bt'), permute(bs,[3 2 1]));
                mod_err(iter,1) = sum(sum(sum(resid.^2)));

                % bv
                resid = bsxfun(@minus,bsxfun(@minus,X,bt'), permute(bs,[3 2 1]));
                bv = sum(sum(resid,2),3)./(NT*NS);
                % bt
                resid = bsxfun(@minus,bsxfun(@minus,X,bv), permute(bs,[3 2 1]));
                bt = sum(sum(resid,1),3)./(NV*NS); bt=bt(:);
                % bs
                resid = bsxfun(@minus,bsxfun(@minus,X,bv),bt');
                bs = sum(sum(resid,1),2)./(NV*NT); bs=bs(:);

                if(iter>1)
                   dev_modl = abs( (mod_err(iter)-mod_err(iter-1))./mod_err(iter-1) );
                end
           end

           % now refine the imputations
           Xest = bsxfun(@plus,bsxfun(@plus,bv,bt'), permute(bs,[3 2 1]));
           % update matrix with imputed values
           X = X.*P + Xest.*(1-P);

           dev_imput = sum(sum( (Xest(P==0) - Xest_old(P==0)).^2 ))/sum(P(:)==0);
           Xest_old = Xest;
       end

    %disp(['ALS terminated. iter# ',num2str(iter),' -- emIter#',num2str(iter_out),' -- dev: ',num2str(dev_modl),' -- err: ',num2str(mod_err(end))])

    out.mod_err = mod_err;
    out.bv = bv;
    out.bt = bt;
    out.bs = bs;

    reco = bsxfun(@plus,bsxfun(@plus,bv,bt'), permute(bs,[3 2 1]));
    out.reco = reco;
    %
    out.impute   = X.*P + out.reco.*(1-P);
    out.merr_av  = sum(sum(sum( (X-out.reco).^2 .*   P  ,1), 2),3)./sum(  P(:)  );
end
