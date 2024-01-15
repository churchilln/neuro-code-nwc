function out = pca_kurtosis_filter( X, kuthr )

[P,N] = size(X); % dimensions
keepindex = 1:N; % tracking input indices
dropindex = [];

% mean-center, svd, kurtosis
X = bsxfun(@minus,X,mean(X,2));
[~,~,v]=svd(X,'econ');
vk1=kurtosis(v(:,1));

kdrop=0; tmpdiff2 = tmpdiff; spm=[];
while(vk1>=kuthr && kdrop<floor(N/2)) %% if excess kurtosis...

    kdrop=kdrop+1;
    
    % index timepoitsn in current matrix by "outlyingness"
    indexx = sortrows( [(1:size(X,2))' abs(v(:,1)-median(v(:,1)))],-2 ); 
    indexx = indexx(:,1);
    
    % iterative deletion until v1 is ~normokurtic
    vk1tmp = vk1;
    iw=0;
    while(iw<size(X,2) && vk1tmp>=kuthr)
         iw=iw+1;
         if(iw==size(X,2))
             error('unable to push v1 to normokurtic by dropping data points');
         else
            vk1tmp = kurtosis( v(indexx(iw+1:end),1) );
         end
    end
    % transfer of indices
    dropindex = [dropindex, keepindex(indexx(1:iw))];
    keepindex( indexx(1:iw) ) = [];
    X(:, indexx(1:iw) )=[];
    % mean-center, svd, kurtosis
    X = bsxfun(@minus,X,mean(X,2));
    [~,~,v]=svd(X,'econ');
    vk1=kurtosis(v(:,1));
    
    ktrace(kdrop,1) = vk1;
end

if( kdrop >= floor(N/2) )
   error('need to discard more than 50% of data to acheive normokurtic!');
end

out.keepindex = keepindex;
out.dropindex = dropindex;
out.ktrace    = ktrace;
   