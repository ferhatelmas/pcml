function e_sqr = sqrerr(T_T, Y)
%SQRERROR: calculates square error for an input vector x
% T_T: 1-K encoding matrix for target vector t
% Y: output vector/matrix for inputs in x/ 3rd level activation matrix for
% K-class MLP
% e_sqr: total error

l2_norms = sum((Y - T_T).^2,1);
e_sqr = 0.5*sum(l2_norms);

