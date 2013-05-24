function err = regerr(X,W,T_T,v)
%REGERR(X,W,T)
% squared error with Tikhonov regularization

Y = W*X;

err = sqerr(T_T,Y) + v*sum(sum(W.^2,2),1);