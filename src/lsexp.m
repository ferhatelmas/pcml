function s = lsexp(y)
%LSEXP(v)
% logsumexp function
% y: input matrix
% s: sum

s = log(sum(exp(y),2));

