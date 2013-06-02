function e_log = logistic_error_multi(X,W,T_T)
%LOGISTIC_ERROR_MULTI(X,W,T_T): calculates logistic error for multi-way classification
% X: target value vectors for input x
% W: weight matrix [wk]'
% T_T: 1-K encoded target matrix

Y = W*X;
e_log = mean(lsexp(Y) - diag(T_T*Y)');

