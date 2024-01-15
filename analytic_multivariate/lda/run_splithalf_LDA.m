function run_splithalf_LDA( input_file1, input_file2, mask_name, onsets, drf, out_prefix )
%
% Single-subject LDA in split-half framework.
%
% Syntax:
%         run_splithalf_LDA( input_file1, input_file2, mask_name, onsets, drf, out_prefix )
%
% Input:
%           input_file1: string specifying name of fMRI run (or split) #1
%           input_file2: string specifying name of fMRI run (or split) #2
%           mask_name  : string giving name of binary brain mask
%
%           onsets: structure with following elements
%
%                 onsets.task1_sp1: timepoints for task1, data split #1
%                 onsets.task2_sp1: timepoints for task2, data split #1
%                 onsets.task1_sp2: timepoints for task1, data split #2
%                 onsets.task2_sp2: timepoints for task2, data split #2
%
%           drf: data reduction fraction (fractional value; 0<drf<1)
%          out_prefix : string giving name of analysis SPM produced as output
%
% Output: single SPM, labelled [ out_prefix,'_LDA_SPM.nii' ]
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
% load run1
V=load_untouch_nii( input_file1 ); 
datamat1 = nifti_to_mat(V,M); 
% load run2
V=load_untouch_nii( input_file2 ); 
datamat2 = nifti_to_mat(V,M); 

% designate output prefix if unspecified
if( isempty(out_prefix) ) out_prefix = ['new']; end

% mean centering
datamat1 = datamat1 - repmat( mean(datamat1,2), [1 size(datamat1,2)]);
datamat2 = datamat2 - repmat( mean(datamat2,2), [1 size(datamat2,2)]);

% create design for split 1
design_sp1 = zeros( size(datamat1,2), 1 );
%
design_sp1(onsets.task1_sp1) = -1;
design_sp1(onsets.task2_sp1) =  1;
% create design for split 2
design_sp2 = zeros( size(datamat2,2), 1 );
%
design_sp2(onsets.task1_sp2) = -1;
design_sp2(onsets.task2_sp2) =  1;

results = lda_optimization ( datamat1, datamat2, design_sp1, design_sp2, drf );

% --- convert to nifti
%
VOL = mask;
VOL(VOL>0) = results.OPT.eig;
%
nii=make_nii( VOL, V.hdr.dime.pixdim(2:4) );
nii.hdr.hist = V.hdr.hist;
nii.hdr.dime.dim(5) = 1;
save_nii(nii,[ out_prefix,'_LDA_SPM.nii' ]); 

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

