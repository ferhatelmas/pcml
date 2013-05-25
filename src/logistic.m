function test_err = logistic(X_L, X_R, T, X_L_val, X_R_val, T_val, X_L_test, X_R_test, T_test, nu, mu)
%LOGISTIC(X_L, X_R, T, X_L_val, X_R_val, T_val, X_L_test, X_R_test, T_test)
% linear classifier with logistic error and gradient descent

k = 5;
[d,n] = size(X_L);
[~,n_test] = size(X_L_test);
[~,n_val] = size(X_L_val);

X = [X_L; X_R; ones(1,n)];
X_test = [X_L_test; X_R_test; ones(1,n_test)];
X_val = [X_L_val; X_R_val; ones(1,n_val)];

T_T = encoder(T);
T_T_test = encoder(T_test);
T_T_val = encoder(T_val);

% training and validation with early-stopping

W = initialize_weights(k,2*d+1);
ec = 0; % epoch count
hdW = W;

while(ec < 1000)
    Y = W*X;
    dW = ((exp(Y-repmat(lsexp(Y),k,1)) - T_T')*X')./n; % calculate gradient
    temp = gradient_descent(hdW, W, dW, nu, mu);
    hdW = W; W = temp;
    tr_err = logerr_multi(X,W,T_T) % training error with updated W
    val_err = logerr_multi(X_val,W,T_T_val); % validation error with updated W
    ec = ec+1;
end

test_err = logerr_multi(X_test,W,T_T_test); % test error with optimized W
