function [ out ] = LinPreClassif( DATA, label, feat, Nout, Nin )
%
%        LINEAR PREDICTIVE CLASSIFIER --> LDA MODEL ONLY
%                                         LOO CROSSVALID
%
%        [ output_args ] = LinPreClassif( DATA, label, model, regul, Nout, Nin )
%
%        DATA = [variables x samples]
%
%

N0 = sum(label==0);
N1 = sum(label==1);

XX0 = DATA(:,label==0);
XX1 = DATA(:,label==1);

%% outer prediction loop
for( no = 1 )
    
   % validation matrices
    XX0_val = XX0; 
    XX1_val = XX1;


%%  %% inner prediction loop
    for( ni = 1:Nin ) 
        
        [no ni],
        
        % get index
        a=randperm(N0); i0_sp1 = a(1:N0-2); i0_sp2 = a(N0-1:end);
        a=randperm(N1); i1_sp1 = a(1:N1-2); i1_sp2 = a(N1-1:end);
        % design matrices
        des_sp1 = [ zeros( length(i0_sp1),1 ); ones( length(i1_sp1),1 ) ];
        des_sp2 = [ zeros( length(i0_sp2),1 ); ones( length(i1_sp2),1 ) ];
        % data matrices
        XX_sp1 = [XX0_val(:,i0_sp1) XX1_val(:,i1_sp1)];
        XX_sp2 = [XX0_val(:,i0_sp2) XX1_val(:,i1_sp2)];
        % model parameters
        parm.avg1 =  mean(XX_sp1,2);
        parm.avg2 =  mean(XX_sp2,2);
        % renormalize, rescale
        XX_sp1_n= bsxfun(@minus,XX_sp1, parm.avg1);
        XX_sp2_n= bsxfun(@minus,XX_sp2, parm.avg2);
        
        if(feat==0)
            
            % run a pca on training data
            [u1 l1]=svd(XX_sp1_n,'econ'); QQ_sp1_n=u1'*XX_sp1_n;
            [u2 l2]=svd(XX_sp2_n,'econ'); QQ_sp2_n=u2'*XX_sp2_n;

            parmset = 1:(size(u1,2)-1);
            
            for(kk=1:length(parmset))            

                % training models:
                LDx = LD_train (QQ_sp1_n(1:parmset(kk),:), des_sp1, 10E-20); LD1 = u1(:,1:parmset(kk))*LDx;
                % prediction
                acc(ni,kk,1) = Classify (LD1, XX_sp1_n, bsxfun(@minus,XX_sp2, parm.avg1), des_sp1, des_sp2);
                %================================
                % training models:
                LDx = LD_train (QQ_sp2_n(1:parmset(kk),:), des_sp2, 10E-20); LD2 = u2(:,1:parmset(kk))*LDx;
                % prediction
                acc(ni,kk,2) = Classify (LD2, XX_sp2_n, bsxfun(@minus,XX_sp1, parm.avg2), des_sp2, des_sp1);
                %
                lds_avg(:,kk,ni) = LD1+LD2;
                lds_dif(:,kk,ni) = LD1-LD2;
            end
            
        elseif(feat==1)
            
            parmset = 10.^(-10:0.33:10);
            
            for(kk=1:length(parmset))
                %================================
                % training models:
                LD1 = LD_train (XX_sp1_n, des_sp1, parmset(kk));
                % prediction
                acc(ni,kk,1) = Classify (LD1, XX_sp1_n, bsxfun(@minus,XX_sp2, parm.avg1), des_sp1, des_sp2);
%                 %================================
%                 % training models:
%                 LD2 = LD_train (XX_sp2_n, des_sp2, parmset(kk));
%                 % prediction
%                 acc(ni,kk,2) = Classify (LD2, XX_sp2_n, bsxfun(@minus,XX_sp1, parm.avg2), des_sp2, des_sp1);
%                 %================================                
%                 %
%                 lds_avg(:,kk,ni) = LD1+LD2;
%                 lds_dif(:,kk,ni) = LD1-LD2;
            end
            
        elseif(feat==2)

            parmset = 10.^(-2:0.33:6);
            
            for(kk=1:length(parmset))
                %================================
                % training models:
                LD1 = svmtrain (XX_sp1_n', des_sp1,'boxconstraint', parmset(kk));
                % prediction
                acc(ni,kk,1) = mean( svmclassify (LD1, bsxfun(@minus,XX_sp2, parm.avg1)') == des_sp2);
                %================================
                % training models:
                LD2 = svmtrain (XX_sp2_n', des_sp2,'boxconstraint', parmset(kk));
                % prediction
                acc(ni,kk,2) = mean( svmclassify (LD2, bsxfun(@minus,XX_sp1, parm.avg2)') == des_sp1);
                %================================                
                %
                lds_avg(:,kk,ni) =0;% LD1+LD2;
                lds_dif(:,kk,ni) =0;% LD1-LD2;
            end            
        end
    end
    
    %LDall(:,:,no) = mean(lds,3);
end

out.acc = mean(acc,3);
% out.LD=mean(lds_avg,3);
% out.LDz=mean(lds_avg,3)./std(lds_avg,0,3);

%% ============================ FUNCTIONS ============================ %%

%%
function LD = LD_train (data, labels, ridge)

% data in full basis
data1 = data (:, labels == 1);
data0 = data (:, labels == 0);
% mean for all bases
mean_dist    = mean (data1, 2) - mean (data0, 2);
within_class = cov (data1') + cov (data0');
LD           = inv (within_class + eye(length(mean_dist))*ridge ) * mean_dist;
LD           = LD ./ sqrt(sum(LD.^2));

%%
function [accuracy] = Classify (LD, train_data, test_data, train_labels, test_labels)

% training means
train_mean0 = mean(train_data(:,train_labels==0),2);
train_mean1 = mean(train_data(:,train_labels==1),2);
% difference from mean in data-space
DIF_0 = bsxfun(@minus, test_data, train_mean0);
DIF_1 = bsxfun(@minus, test_data, train_mean1);
% basis x scan
log_pp0 = -0.5*(DIF_0'*LD).^2;
log_pp1 = -0.5*(DIF_1'*LD).^2;

accuracy = mean( double(double( log_pp1>log_pp0 ) == test_labels) );



%%
%%

% 
% % do not standardize the variables prior to analysis
% standardize = false;
% % dataset
% T_class = sign( labels - 0.5 );
% % LARS-LASSO - show the full trace of explored solutions
% [Beta, A, mu, C, c, gamma] = LARS_flex(data', T_class(:), 'lars', Inf, standardize, Range+1);
% % turn into col-vects of Beta weights - discard bias weights
%  Beta = Beta(2:end,:)';
% 
% % % %% Currently unused: for 'lasso' flag, discard degenerate solutions
% % % numVar   = ( sum(Beta~=0) );
% % % transIdx = [1 numVar(2:end)-numVar(1:end-1)];
% % % % keep only steps where new variable added
% % % Beta = Beta(:,transIdx>0);
% 
% % keep only 1..Range steps
% LD.lin_discr = Beta(:,1:Range);