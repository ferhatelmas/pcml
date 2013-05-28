function err = regerr(X,W,T_T,v)
%REGERR(X,W,T)
% squared error with Tikhonov regularization

Y = X*W;
err = sqrerr(T_T',Y') + v*sum(mean(W.^2));