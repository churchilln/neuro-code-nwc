function mip_3way( file, clustr, axes )
%[59 55 38]
%[50 57 42]

[p n e] = fileparts(file);

V   = load_untouch_nii( file );
vol = double(V.img);

if( ~isempty(clustr) && clustr>0 )
vol = clust_up( vol, clustr );
end

% midD = round(size(vol)./2);
if(isempty(axes)) midD = round(size(vol)./2);
else              midD = axes;
end

if( sum( vol(:)>eps )>10 )

%% ---
blank=zeros(size(vol)); 
blank(midD(1),:,:,1) = max(vol,[],1);
blank(:,midD(2),:,2) = max(vol,[],2);
blank(:,:,midD(3),3) = max(vol,[],3);

V.img = blank;
V.hdr.dime.dim(5)=3;
save_untouch_nii(V,[n,'_mip+',e]);

end
if( sum( vol(:)<-eps )>10 )

%% ---
blank=zeros(size(vol));
blank(midD(1),:,:,1) = min(vol,[],1);
blank(:,midD(2),:,2) = min(vol,[],2);
blank(:,:,midD(3),3) = min(vol,[],3);

V.img = blank;
V.hdr.dime.dim(5)=3;
save_untouch_nii(V,[n,'_mip-',e]);

end