function out = imputation_svd_modeltest( X, str, Ndel, inloop, outloop,temp_name )
%
% SYNTAX:
%
%        out = imputation_svd_modeltest( X, str, Ndel, inloop, outloop, temp_name )
%
% INPUT:
%
%          X : input data. must be either a (subj x time) or a (subj x time x vox) array
%        str : special data structure (submean, subvar, type, Lam)
%       Ndel : number of datapoints to delete ("simulate as missing")
%     inloop : number of inner-loop iterations; rerun each time with a new set of "sim missing"
%    outloop : number of outer-loop iterations, re-running whole inner loop this many times to assess stability of optima (only available on 2d arrays)
%  temp_name : string array. if non-empty, saves to file with this suffix
%
%
%
%
%
%

if(nargin<6) temp_name=''; end


if( size(X,3)==1 ) % 2D matrix

    % indexing all existing values (i.e., finite ones)
    isfin  = find( isfinite(X)); 

    for(tt=1:outloop) %% run multiple replicates to assess stability
        for(ii=1:inloop)
            [tt tt tt ii ii ii],
            % randomized selection of existing values --> will change to missing
            risnfin = isfin( randperm(numel(isfin),Ndel) );
            [rix rjx]=ind2sub( size(X), risnfin );
            % create "simulated-missing" matrix, then run imputation
            X_sm         =   X; 
            X_sm(risnfin)= NaN;
            X_est = imputation_svd( X_sm, str );
            % evaluate model error
            merr(ii,:) = sum(sum( bsxfun(@minus,X,X_est).^2,1,'omitnan'),2,'omitnan') ./ Ndel;   
            % storing model error values
            for(jj=1:numel(risnfin))
                ervl(ii,jj,:) = X_est(rix(jj),rjx(jj),:)-X(rix(jj),rjx(jj));
            end
        end
        [out.optvalset(tt,1), out.optixset(tt,1)] = min( mean(merr,1),[],2 ); % L minimizing mean error
        merrset(:,:,tt)   = merr;
        ervlset(:,:,:,tt) = ervl;
    end
    out.optval = median(out.optvalset);
    out.optix  = mode(out.optixset );
    out.merr   = permute( merrset, [2 1 3]);   % lam x inloop x outloop
    out.ervl   = permute( ervlset, [3 1 2 4]); % lam x inloop x n x outloop
    
else % 3D array
    
    [ns,nt,nv] = size(X);
    % indexing all coordinates with existing values - over all voxels!
    [i,j]  = find( isfinite(mean(X,3))); 
    isfin_sub  = [i,j]; % concat indices to matrix
    % now matricize the X data array
    X_resh = reshape( permute(X,[1 3 2]), size(X,1),[] ); % subj x ( Nvox*time1, Nvox*time2, ... )

    for(ii=1:inloop)
        [ii ii ii ii ii],
        % randomized selection of existing values --> will be changed to missing
        risnfin_sub = isfin_sub( randperm(size(isfin_sub,1),Ndel),: );
        % create "simulated-missing" matrix, then run imputation
        X_resh_sm =  X_resh; 
        for(k=1:size(risnfin_sub,1)) % ith subject, jth block of size nv
            X_resh_sm( risnfin_sub(k,1),(nv*(risnfin_sub(k,2)-1) + 1) : nv*risnfin_sub(k,2) ) = NaN;
        end
        X_resh_est = imputation_svd( X_resh_sm, str );
        % evaluate model error
        out.merr(ii,:) = sum(sum( bsxfun(@minus,X_resh,X_resh_est).^2,1,'omitnan'),2,'omitnan') ./ (Ndel*nv);
        save(['merr_tmp_',temp_name,'.mat'],'out');
    end
    [out.optval, out.optix] = min( mean(out.merr,1),[],2 ); % L minimizing mean error
    
end