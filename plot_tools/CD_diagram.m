function [ output ] = CD_diagram( respBlock, format, labels )
%
% =========================================================================
% CD_DIAGRAM: tool for plotting critical-difference diagrams, obtained from
% post-hoc evaluation of Friedman test; this is the non-parametric
% equivalent of a repeat-measures ANOVA.
% =========================================================================
%
% Input:
%
%      respBlock = matrix of "responses" for a set of treatments/experimental 
%                  manipulations. The format is (treatments x observations), 
%                  e.g.  each column is a set of treatment responses, 
%                  measured in a single sample (subject/dataset/etc...)
%
%      format    = string specifying the plot display format. Two options:
%
%                    'small': ideal for a small number of treatments (e.g. <10).
%                    'large': ideal for a large number of treatments (e.g. >>10)
%
%      labels    = cell array of strings, labelling each of the treatment types
%                     e.g. labels = {'treat1','treat2'...}
%                  If this is empty, the treatments will just be numbered
%                  in their original matrix ordering
%
%
% Output: a single plot, plus and "output" structure containing summary results
%-----------------------
% 1. Output structure:
%
%    output.mean_ranks  = the average ranking of each treatment
%    output.Fprob       = probability of significant rank-order, based on Friedman test
%    output.CD          = critical-difference interval (at alpha=.05)
%-----------------------
% 2a. Plot, type='small'
%
%    - horizontal grey bar denotes the range of possible ranks
%    - vertical black bars are average ranks of the different treatments (labelled)
%    - horizontal black bar is the critical difference (CD) interval; 
%      treatments separated by less than CD are not statistically distinguishable
%    - horizontal blue bar covers the optimal (highest-ranked) set of treatments, 
%      that are not statistically different
%    - horizontal red bar covers the worst (lowest-ranked) set of treatments, 
%      that are not statistically different
%-----------------------
% 2b. Plot, type='large'
%
%    - horizontal bars are the average ranks of the different treatments (labelled)
%    - horizontal black bar (bottom) is the critical difference (CD) interval; 
%      treatments separated by less than CD are not statistically distinguishable
%    - blue bars include the optimal (highest-ranked) set of treatments, 
%      that are not statistically different
%    - red bars include the worst (lowest-ranked) set of treatments, 
%      that are not statistically different
%-----------------------
%    IMPORTANT NOTES:
%
%    * we display Friedman test significance; if non-significant (p>0.10),
%      you CANNOT do post-hoc testing (as the rankings are not stable)
%    * this plot assumes higher rank = "better" response; if this is not
%      the case, the colours just mean the opposite (e.g. red is now "best")
%
%    * These statistics are only exact for independent observations
%    * test statistics given by Chi-Square approximation; ideal performance 
%      is given for large samples and/or many treatments
%
% ------------------------------------------------------------------------%
%  Author: Nathan Churchill, University of Toronto
%  email:  nathan.churchill@rotman.baycrest.on.ca
% ------------------------------------------------------------------------%
% version history: October 25 2013
% ------------------------------------------------------------------------%


% fixed post-hoc testing significance of .05
alpha = 0.05;
% perform friedman rank-sum test
[ prob, sigdiff ] = friedman_test_ex( respBlock, alpha );
% some parameters
s         = length( respBlock(:,1) ); % number of treatments
rankBlock = tiedrank( respBlock );    % rank treatment for each observed set
meanRank  = mean( rankBlock, 2 );     % compute mean rank, per treatment
% create labels, if not provided
if(isempty(labels))
   for(i=1:s) 
        labels{i} = num2str(i);
   end
end

% output structure
output.mean_ranks = meanRank;
output.Fprob = prob;
output.CD    = sigdiff;

% plotting
if( strcmp( format, 'small' ) ) %% for small numbers of treatments

    figure; hold on;
    % plot the mean ranks
    plot( [1 s], [0 0], 'linewidth', 4, 'color', [0.4 0.4 0.4] );
    for(i=1:s)
        plot( meanRank(i)*[1 1], 0.1*s.*[-1 1], '-k', 'linewidth',2 );
        text( meanRank(i), 0.15*s, labels{i} );
    end
    % display Friedman significance
    if    ( prob < 0.01 ) text( 0.25,  0.25*s, ['Friedman test p=',num2str(prob),' (significant)']);
    elseif( prob < 0.10 ) text( 0.25,  0.25*s, ['Friedman test p=',num2str(prob),' (marginal)']);
    else                  text( 0.25,  0.25*s, ['Friedman test p=',num2str(prob),' (non-significant)']);
    end
    % plot critical difference
    plot( 1+[0 sigdiff], -0.2.*s.*[1 1], '-k', 'linewidth',2 );
    text( 1, -0.25*s, 'critical difference');
    % find best/worst subgroups based on CD interval
    topRanks = meanRank( meanRank >= ( max(meanRank) - sigdiff ) );
    lowRanks = meanRank( meanRank <= ( min(meanRank) + sigdiff ) );

    % plot cd-connector bars
    if( prob < 0.10 )
       %
       % lowest-ranked
       plot( [ min(meanRank) max(lowRanks) ], 0.110*s.*[1 1], '-r' );
       plot( [ min(meanRank) max(lowRanks) ], 0.115*s.*[1 1], '-r' );
       % highest-ranked
       plot( [ max(meanRank) min(topRanks) ], 0.110*s.*[1 1], '-b' );
       plot( [ max(meanRank) min(topRanks) ], 0.115*s.*[1 1], '-b' );
    end
    
    xlabel('Mean Ranking');
    set(gca,'YTickLabel',[]);
    
    xlim([0.01 s+0.99]);
    ylim([-0.5*s 0.5*s]);
    
elseif( strcmp( format, 'large' ) ) %% for large numbers of treatments
    
    % reorder the treatments low-rank --> high-rank (worst --> best)
	sortidx   = sortrows([(1:s)' meanRank],-2);
    sortidx   = sortidx(:,1);
    meanRank2 = meanRank(sortidx);
    labels2   = cell( length(labels)+1,1 );
    labels2{1} = 'Crit.Diff.';
    meanRank2 = [sigdiff; meanRank2];
    for(i=1:s) labels2{i+1} = labels{sortidx(i)}; end
    
    
    figure; hold on;
    % overlay CD bar
    barh( 1:s+1, meanRank2, 0.25,'k' );
    % overlay high-ranked bars
    meanRankTemp = meanRank2;
    meanRankTemp(1) = 0;
    barh( 1:s+1, meanRankTemp, 0.50,'facecolor','b', 'linewidth',2 );
    % overlay middle bars
    meanRankTemp = meanRank2;
    meanRankTemp( meanRankTemp > max(meanRank) - sigdiff ) = 0;
    meanRankTemp(1) = 0;
    barh( 1:s+1, meanRankTemp, 0.50,'facecolor',[0.5 0.5 0.5], 'linewidth',2 );
    % overlay low-ranked bars
    meanRankTemp = meanRank2;
    meanRankTemp( meanRankTemp > min(meanRank) + sigdiff ) = 0;
    meanRankTemp(1) = 0;
    barh( 1:s+1, meanRankTemp, 0.50,'facecolor','r', 'linewidth',2 );
    
    set(gca,'YTickLabel',labels2);
    xlabel('Mean Ranking');

    % display Friedman probability
    if    ( prob < 0.01 ) text( 0.1,  s+1.6, ['Friedman test p=',num2str(prob),' (significant)']);
    elseif( prob < 0.10 ) text( 0.1,  s+1.6, ['Friedman test p=',num2str(prob),' (marginal)']);
    else                  text( 0.1,  s+1.6, ['Friedman test p=',num2str(prob),' (non-significant)']);
    end
    ylim([0.1 s+1.9]);
    xlim([-0.5 s+0.9]);
else
    %
    error('Need to specify format - either small or large!');
end


%%
function [ prob, sigdiff ] = friedman_test_ex( respBlock, varargin )
% performs the Friedman test statistic on multiple-treatment blocks:
% 
%         [prob sigdiff] =  friedman_test( respBlock, (alpha) ) 
%
% wherein [ respBlock = 'treatments' x 'observations' ]
% e.g. each column is a block of observed treatments.
% 
% * Note that observations should be independant
% * Note also: test statistic given by Chi-Square approximation;
%   ideal performance is given for large samples and/or many treatments
% * sigdiff = critical difference at given alpha, for test
% * if alpha not specified, default is 0.05

if( nargin == 2 )
    alpha = varargin{1};
else
    alpha = 0.05;
end

N = length( respBlock(1,:) ); % no. cols = no. observations
s = length( respBlock(:,1) ); % no. rows = no. treatments

% take observation blocks and rank treatments in each:
rankBlock = tiedrank( respBlock ); % rank treatment for each observed set
meanRank  = mean( rankBlock, 2 );          % compute mean rank, per treatment
grandMean = (s+1)/2;                       % compute grand mean on ranks:

% get friedman statistic:
% sum-of-squares difference between all treatment means and grand mean,
% with factor giving normal approximation for large samples:
Q = 12*N/( s*(s+1) ) * sum( (meanRank - grandMean).^2 );

% probability of difference >= Q approximated by the Chi-Square 
% cumulative distribution (for df = s-1):
prob   = 1 - chi2cdf( Q, s-1 );

% computing critical difference:
% (1) set critical inv-tscore
thresh = tinv(1-alpha/2, (N-1)*(s-1));
% (2) compute C-value, based on design
C = N*s*(s+1)*(s-1)/12;
% (3) now compute critical difference in ranks for sig. difference
sigdiff = (thresh/N) * sqrt( ( 2*N*C/( (N-1)*(s-1)) ) * (1 - Q/(N*(s-1))) );
