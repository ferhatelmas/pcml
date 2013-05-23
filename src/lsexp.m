function s = lsexp(v)
%LSEXP(v)
% logsumexp function
% v: input vector
% s: sum

s = log(sum(exp(v)));

