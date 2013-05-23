function e_sqr = sqrerr(t, a_3)
%SQRERROR: calculates square error for an input vector x
% t: target value vectors for input x
% a_3: MLP 3rd activation level output vector/matrix for inputs in x
% e_sqr: total error

k = size(a_3,1); % number of targets
n = size(a_3,2); % number of input samples

% build 1-of-K encoding matrix of targets
li = 0:k:(n-1)*k;
t_t = zeros(k,n);
t_t(li+t) = 1;

l2_norms = sum((a_3 - t_t).^2,1);
e_sqr = 0.5*sum(l2_norms);

