function [ out ] = LinPreClassif( DATA, label, feat, Nout, Nin )
%
%        LINEAR PREDICTIVE CLASSIFIER --> CAN DO MANY DIFFERENT MODELS!
%                                         2-LOOP K-FOLD CROSSVALID
%
%        [ output_args ] = LinPreClassif( DATA, label, model, regul, Nout, Nin )
%
%        DATA = [variables x samples]
%
%

% datapoints per class
N0 = sum(label==0);
N1 = sum(label==1);
% split matrices
XX0 = DATA(:,label==0);
XX1 = DATA(:,label==1);
% folding (held out per class)
nval = 5;
ntst = 5;

%% outer prediction loop
for( no = 1:Nout )
    
    % get index
    a=randperm(N0); i0=a(1:ntst);
    a=randperm(N1); i1=a(1:ntst);
    % validation matrices
    XX0_val = XX0; XX0_val(:,i0)=[];
    XX1_val = XX1; XX1_val(:,i1)=[];
    % test matrix
    XX_test = [XX0(:,i0) XX1(:,i1)];
    
    % for valid. after
    XX_sp0 = [XX0_val XX1_val];
    % model parameters
    parm0.avg1 =  mean(XX_sp0,2);
    % renormalize, rescale
    XX_sp0_n= bsxfun(@minus,XX_sp0, parm0.avg1);
    des_sp0 = [ zeros( N0-ntst,1 ); ones( N1-ntst,1 ) ];
  
    %% UNREGULARIZED MODELS -- OUTERLOOP ONLY!
    if( strcmp(feat,'GNB') || strcmp(feat,'PLS') )
        acc=0;
        if(strcmp(feat,'GNB'))
            'g',
            test(no,1) = do_gnb( XX0_val, XX1_val, XX0(:,i0), XX1(:,i1),'linear');
        elseif(strcmp(feat,'PLS'))
            error('not yet.');
        end
        
    %% REGULARIZED MODELS -- INNER VALIDATION LOOP FOR TUNING
    else
        for( ni = 1:Nin ) 

            [no ni],

            % get index
            a=randperm(N0-1); i0_sp1 = a(1:end-nval); i0_sp2 = a(end-nval+1:end);
            a=randperm(N1-1); i1_sp1 = a(1:end-nval); i1_sp2 = a(end-nval+1:end);
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

            if(strcmp(feat,'LD-PCA'))

                % run a pca on training data
                [u1 l1]=svd(XX_sp1_n,'econ'); QQ_sp1_n=u1'*XX_sp1_n;

                parmset = 1:(size(u1,2)-1);

                for(kk=1:length(parmset))            

                    % training models:
                    LDx = LD_train (QQ_sp1_n(1:parmset(kk),:), des_sp1, 10^-6);
                    % transform into voxel space
                    LD  = u1(:,1:parmset(kk))*LDx;
                    % prediction
                    acc(ni,kk,no) = Classify_LD (LD, XX_sp1_n, bsxfun(@minus,XX_sp2, parm.avg1), des_sp1, des_sp2);
                    lds(:,kk,ni)  = LD;
                end

            elseif(strcmp(feat,'LD-L2') || strcmp(feat,'ridge'))

                parmset = 10.^(-9:0.25:9);

                for(kk=1:length(parmset))            
                    % training models:
                    LD = LD_train (XX_sp1_n, des_sp1, parmset(kk));
                    % prediction
                    acc(ni,kk,no) = Classify_LD (LD, XX_sp1_n, bsxfun(@minus,XX_sp2, parm.avg1), des_sp1, des_sp2);
                    lds(:,kk,ni)  = LD;
                end
            elseif(strcmp(feat,'SVM'))

                parmset = 10.^(-2:0.33:6);

                for(kk=1:length(parmset))
                    % training models:
                    LD = svmtrain (XX_sp1_n', des_sp1,'boxconstraint', parmset(kk));
                    % prediction
                    acc(ni,kk,no) = mean( svmclassify (LD, bsxfun(@minus,XX_sp2, parm.avg1)') == des_sp2);
                end                        
            end
        end
        % maximized prediction
        [vx ix] = max( mean(acc(:,:,no),1) );
        % held-out sample prediction
        if(feat==1)
        parmset = 10.^(-9:0.25:9);
        LD = LD_train (XX_sp0_n, des_sp0, parmset(ix));
        test(no,1) = Classify_LD (LD, XX_sp0_n, bsxfun(@minus,XX_test, parm0.avg1), des_sp1, [zeros(ntst,1); ones(ntst,1)]);   
        elseif(feat==2)
        parmset = 10.^(-2:0.33:6);
        LD = svmtrain (XX_sp0_n, des_sp0,'boxconstraint', parmset(ix));
        test(no,1) = mean( svmclassify (LD, bsxfun(@minus,XX_test, parm0.avg1)') == [zeros(ntst,1); ones(ntst,1)]);
        end 
    end
end

out.acc = acc;
out.test = mean(test);
% out.LD=mean(LDall,3);
% out.LDz=mean(LDall,3)./std(LDall,0,3);

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
function [accuracy] = Classify_LD (LD, train_data, test_data, train_labels, test_labels)

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
function PP = do_gnb( trnMat1,trnMat2, tstMat1,tstMat2, decision_model )
% 

if( strcmp( decision_model, 'linear' ) )
    
    % * computed pooled variance statistic between the two classes

    % compute sample means and variances for train/test
    avg1_trn = mean(trnMat1,    2);
    avg2_trn = mean(trnMat2,    2);
    % compute number of samples (image-vectors) for each group
    n1_tst = size (tstMat1,  2);
    n2_tst = size (tstMat2, 2);      
    
    % preparatory: get the log-pp at each voxel / class ... done for full data matrix
    CORR_sum = 0;
    %
    for(w=1:n1_tst)
        logPP_t = - ((tstMat1(:,w) - avg1_trn).^2); logPP_t(~isfinite(logPP_t)) =0;
        logPP_f = - ((tstMat1(:,w) - avg2_trn).^2); logPP_f(~isfinite(logPP_f)) =0;
        % add to "correct" sum: weight=(1 for correct / 0.5 for undecided)
        CORR_sum  = CORR_sum + double( sum(logPP_t) >= sum(logPP_f) );
    end
    %
    for(w=1:n2_tst)
        logPP_t = - ((tstMat2(:,w) - avg2_trn).^2); logPP_t(~isfinite(logPP_t)) =0;
        logPP_f = - ((tstMat2(:,w) - avg1_trn).^2); logPP_f(~isfinite(logPP_f)) =0;
        % add to "correct" sum: weight=(1 for correct / 0.5 for undecided)
        CORR_sum  = CORR_sum + double( sum(logPP_t) >= sum(logPP_f) );
    end
    % classifier accuracy:
    PP = CORR_sum ./ (n1_tst+n2_tst);   
    
elseif( strcmp( decision_model, 'nonlinear' ) )
    
    % * based on the class-specific variance estimates

    % compute sample means and variances for train/test
    avg1_trn = mean(trnMat1,    2);
    avg2_trn = mean(trnMat2,    2);
    std1_trn = std (trnMat1, 0, 2);
    std2_trn = std (trnMat2, 0, 2);
    % compute number of samples (image-vectors) for each group
    n1_trn = size (trnMat1, 2);       n1_tst = size (tstMat1,  2);
    n2_trn = size (trnMat2, 2);       n2_tst = size (tstMat2, 2);      
    
    % preparatory: get the log-pp at each voxel / class ... done for full data matrix
    CORR_sum = 0;
    %
    for(w=1:n1_tst)
        logPP_t = -log(std1_trn) - ((tstMat1(:,w) - avg1_trn).^2)./(2*std1_trn.^2); logPP_t(~isfinite(logPP_t)) =0;
        logPP_f = -log(std2_trn) - ((tstMat1(:,w) - avg2_trn).^2)./(2*std2_trn.^2); logPP_f(~isfinite(logPP_f)) =0;
        % add to "correct" sum: weight=(1 for correct / 0.5 for undecided)
        CORR_sum  = CORR_sum + double( sum(logPP_t) > sum(logPP_f) );
    end
    %
    for(w=1:n2_tst)
        logPP_t = -log(std2_trn) - ((tstMat2(:,w) - avg2_trn).^2)./(2*std2_trn.^2); logPP_t(~isfinite(logPP_t)) =0;
        logPP_f = -log(std1_trn) - ((tstMat2(:,w) - avg1_trn).^2)./(2*std1_trn.^2); logPP_f(~isfinite(logPP_f)) =0;
        % add to "correct" sum: weight=(1 for correct / 0.5 for undecided)
        CORR_sum  = CORR_sum + double( sum(logPP_t) > sum(logPP_f) );
    end
    % classifier accuracy:
    PP = CORR_sum ./ (n1_tst+n2_tst);   
end



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