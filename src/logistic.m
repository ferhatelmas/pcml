function error = logistic(X_L, X_R, T, X_L_val, X_R_val, T_val, X_L_test, X_R_test, T_test, nu, mu)
%LOGISTIC(X_L, X_R, T, X_L_val, X_R_val, T_val, X_L_test, X_R_test, T_test)
% linear classifier with logistic error and gradient descent

[d,n] = size(X_L);
[~,n_test] = size(X_L_test);
[~,n_val] = size(X_L_val);

X = [X_L; X_R; ones(1,n)]';
X_test = [X_L_test; X_R_test; ones(1,n_test)]';
X_test = [X_L_val; X_R_val; ones(1,n_val)]';

T_T = encoder(T);
T_T_test = encoder(T_test);
T_T_val = encoder(T_val);

% training and validation with early-stopping

W = initialize_weights(2*n+1,k);
ec = 0; % epoch count
hdW = W; % previous W

while(ec < 100)
    Y = X*W;
    dW = exp(Y-repmat(lsexp(Y),n,1)) - T_T; % calculate gradient
    temp = gradient_descent(hdw, W, dW, nu, mu);
    hdW = W; W = temp;
    tr_err = logerr_multi(X_tr,W,T_T); % training error with updated W
    val_err = logerr_multi(X_val,W,T_T_val); % validation error with updated W
end

test_err = logerr_multi(X_test,W,T_T_test); % test error with optimized W
