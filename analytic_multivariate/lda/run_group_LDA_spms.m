function run_group_LDA_spms( input_list, mask_name, labels, drf, out_prefix )
%
%
% Group-level LDA in split-half framework.
%
% Syntax:
%         run_group_LDA( input_list, mask_name, labels, drf, out_prefix )
%
% Input:
%           input_list:  cell array, where each entry gives name
%                        of a single fMRI run, for a different subject
%                        e.g.
%                             input_list{1} = 'directory1/subject1_spm.nii'
%                             input_list{2} = 'directory2/subject2_spm.nii'
%                             ...
%           mask_name  : string giving name of binary brain mask
%
%           labels: binary vector, where each entry gives label of
%                   corresponding subject
%                   e.g.
%                       labels(1): label for subject#1
%                       labels(2): label for subject#2
%                       ...
%
%           drf: data reduction fraction (fractional value; 0<drf<1)
%          out_prefix : string giving name of analysis SPM produced as output
%
% Output: single SPM, labelled [ out_prefix,'_LDA_SPM_group.nii' ]
%
% ------------------------------------------------------------------------%
%
%  This code was developed by Nathan Churchill Ph.D., University of Toronto,
%  Email: nchurchill@research.baycrest.org
%
% ------------------------------------------------------------------------%
% version history: 2013/07/21
% ------------------------------------------------------------------------%


% load mask volume
M=load_untouch_nii( mask_name );
mask     = double(M.img);

N_subject= length(input_list);

for(n=1:N_subject)

    % load run n
    V=load_untouch_nii( input_list{n} ); 
    datamat(:,n) = nifti_to_mat(V,M);
end
 
labels=labels(:);
if(numel(unique(labels))~=2) error('label vector should have exactly 2 values'); end
labels = sign( double(labels==max(labels))-0.5 );
datamat0 = datamat(:,labels<0); n0 = size(datamat0,2);
datamat1 = datamat(:,labels>0); n1 = size(datamat1,2);

% designate output prefix if unspecified
if( isempty(out_prefix) ) out_prefix = ['new']; end

eig_set=0;

for(iter=1:50)
    disp(['iter#',num2str(iter)]);
    list0 = randperm(n0);
    list1 = randperm(n1);
    %
    data_sp1 = [datamat0(:,list0(1:round(n0/2))    ) datamat1(:,list1(1:round(n1/2))    )];
    data_sp2 = [datamat0(:,list0(round(n0/2)+1:end)) datamat1(:,list1(round(n1/2)+1:end))];
    %
    design_sp1 = [-ones(round(n0/2),1); ones(round(n1/2),1)];
    design_sp2 = [-ones(n0-round(n0/2),1); ones(n1-round(n1/2),1)];
    %
    res = lda_optimization ( data_sp1, data_sp2, design_sp1, design_sp2, drf );

    r_set(:,iter) = res.R;
    p_set(:,iter) = res.P;
    eig_set       = eig_set + res.eig./50;
  
end

optstr = {'Dmin','Rmax','Pmax'};
  
% quick optimization on D metric:
[vx ix] = min( sqrt( (1-median(r_set,2)).^2 + (1-median(p_set,2)).^2 ) ); 
    ixset(1,1) = ix(1);
[vx ix] = max( median(r_set,2) ); 
    ixset(1,2) = ix(1);
[vx ix] = max( median(p_set,2) ); 
    ixset(1,3) = ix(1);

eigopt = eig_set(:,ixset);

for(i=1:3)
    disp(['Optimization based on ',optstr{i},'...']);
    disp(['   ...reproducibility: ',num2str(median(r_set(ixset(i),:)))]);
    disp(['   ...prediction:      ',num2str(median(p_set(ixset(i),:)))]);
    
    % --- convert to nifti
    %
    VOL = mask;
    VOL(VOL>0) = eigopt(:,i);
    %
    nii=make_nii( VOL, V.hdr.dime.pixdim(2:4) );
    nii.hdr.hist = V.hdr.hist;
    nii.hdr.dime.dim(5) = 1;
    save_nii(nii,[ out_prefix,'_LDA_SPM_group_',optstr{i},'.nii' ]);     
end


    
%%
function dataMat = nifti_to_mat( niiVol, niiMask )
%
% take niiVol and niiMask (nifti) format, and convert
% to matlab vector/matrix structure:
%
% dataMat = nifti_to_mat( niiVol, niiMask )
%
vol = double(niiVol.img);
msk = double(niiMask.img);

dataMat = zeros( sum(msk(:)>0), size(vol,4) );

for(t=1:size(vol,4))
    tmp=vol(:,:,:,t);
    dataMat(:,t) = tmp(msk>0);
end

