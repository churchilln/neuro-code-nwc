function run_group_LDA( input_list, mask_name, onsets, drf, out_prefix )
%
%
% Group-level LDA in split-half framework.
%
% Syntax:
%         run_group_LDA( input_list, mask_name, onsets, drf, out_prefix )
%
% Input:
%           input_list:  cell array, where each entry gives name
%                        of a single fMRI run, for a different subject
%                        e.g.
%                             input_list{1} = 'directory1/subject1_data.nii'
%                             input_list{2} = 'directory2/subject2_data.nii'
%                             ...
%           mask_name  : string giving name of binary brain mask
%
%           onsets: cell array, where each entry corresponds to task onsets
%                   for a given subject. 
%                   e.g.
%                       onsets{1}.task1: timepoints for task 1, subject#1
%                       onsets{1}.task2: timepoints for task 2, subject#1
%
%                       onsets{2}.task1: timepoints for task 1, subject#2
%                       onsets{2}.task2: timepoints for task 2, subject#2
%
%                       ....
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
    datamat = nifti_to_mat(V,M);
    datamat = datamat - repmat( mean(datamat,2), [1 size(datamat,2)]);
    %
    datamat_task1{n} = datamat(:, onsets{n}.task1);
    datamat_task2{n} = datamat(:, onsets{n}.task2);
end
    

% designate output prefix if unspecified
if( isempty(out_prefix) ) out_prefix = ['new']; end

results = lda_optimization_group ( datamat_task1, datamat_task2, drf, 50 );

% --- convert to nifti
%
VOL = mask;
VOL(VOL>0) = results.OPT.eig;
%
nii=make_nii( VOL, V.hdr.dime.pixdim(2:4) );
nii.hdr.hist = V.hdr.hist;
nii.hdr.dime.dim(5) = 1;
save_nii(nii,[ out_prefix,'_LDA_SPM_group.nii' ]); 
    
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

