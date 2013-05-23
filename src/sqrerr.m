function e_sqr = sqrerr(t, a_3)
%SQRERROR: calculates square error for an input vector x
% t: target value vectors for input x
% a_3: MLP 3rd activation level output vector/matrix for inputs in x
% e_sqr: total error

[k,~] = size(a_3); % number of targets and samples

T_T = encoder(t,k);

l2_norms = sum((a_3 - T_T).^2,1);
e_sqr = 0.5*sum(l2_norms);

