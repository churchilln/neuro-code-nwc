function [ prob, sigdiff ] = quade( respBlock, varargin )
% performs the Friedman test statistic on multiple-treatment blocks:
% 
%         [prob sigdiff] =  friedman( respBlock, (alpha) ) 
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
% rank treatments for each sample
R = tiedrank( respBlock );
%Compute the range of each block and then rank them.
Q=tiedrank(range(respBlock,2)); 
%Compute a modified version of the Friedman matrix
%Note that the row sums of the matrix T are all 0 (and, of course, the sum of the column sums)...
T=(R-(N+1)/2).*repmat(Q,1,N);
T2=sum(sum(T).^2)/s;
k=s-1;
W=(k*T2)/(sum(sum(T.^2))-T2); %The Quade statistic.
%The Quade statistic is approximanble with the F distribution.
dfn=N-1;
dfd=dfn*k;
prob=1-fcdf(W,dfn,dfd);
% computing critical difference:
% (1) set critical inv-tscore
thresh = tinv(1-alpha/2, (N-1)*(s-1));
% (2) compute C-value, based on design
C = N*s*(s+1)*(s-1)/12;
% (3) now compute critical difference in ranks for sig. difference
sigdiff = (thresh/N) * sqrt( ( 2*N*C/( (N-1)*(s-1)) ) * (1 - W/(N*(s-1))) );
