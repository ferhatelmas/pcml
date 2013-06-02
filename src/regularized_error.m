function err = regularized_error(X, W, T_T, v)
%REGULARIZED_ERROR(X,W,T)
% squared error with Tikhonov regularization

Y = X*W;
err = squared_error(T_T',Y') + v*sum(mean(W.^2));