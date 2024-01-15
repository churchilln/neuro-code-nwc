function map = LR_map (LR, basis)
% create a sensitivity map for Fisher's linear discriminant
% INPUTS:
% LD -- LD structure (created by LD_create)
% basis -- PC basis (#voxels x #PCs)
% OUTPUT: sensitivity map (#voxels x 1)

map = basis * LR.lin_discr;