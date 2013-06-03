function [tr_err, val_err, zero_one_err, test_err, zero_one_err_test, confused] = logistic_regression(X_L, X_R, T, X_L_val, X_R_val, T_val, X_L_test, X_R_test, T_test, nu, mu)
%LOGISTIC_REGRESSION(X_L, X_R, T, X_L_val, X_R_val, T_val, X_L_test, X_R_test, T_test)
% linear classifier with logistic error and gradient descent
% CALL PREPARE_MULTI BEFORE RUNNING

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
max_ec = 1000; % maximum epoch count allowed
hdW = W;
tr_err = zeros(1,max_ec);
val_err = zeros(1,max_ec);
zero_one_err = zeros(1,max_ec);

early_stop = ones(1,2); % stores last two differences btw consecutive zero_one errors
es = 0;
prev_err = 0;

% stop when three same or very close error values occur 3 in a row
while(sum(early_stop > 1e-6*ones(1,2)) > 0)
    es = mod(es + 1,2);
    ec = ec+1
    Y = W*X;
    dW = ((exp(Y-repmat(lsexp(Y),k,1)) - T_T')*X')./n; % calculate gradient
    temp = gradient_descent(hdW, W, dW, nu, mu);
    hdW = W; W = temp;
    tr_err(ec) = logistic_error_multi(X,W,T_T); % training error with updated W
    val_err(ec) = logistic_error_multi(X_val,W,T_T_val); % validation error with updated W
    [~,c] = max(W*X_val,[],1); % find index of maximum among each sample output
    zero_one_err(ec) = mean(c-1 ~= T_val);
    early_stop(es+1) = abs(zero_one_err(ec) - prev_err);
    prev_err = zero_one_err(ec);
    plotter(tr_err, val_err, zero_one_err, ec);
end

test_err = logistic_error_multi(X_test,W,T_T_test); % test error with optimized W

% zero-one error
[~,c] = max(a_3,[],1); % find index of maximum among each sample output
zero_one_err_test = mean(c-1 ~= T_test);
confused = confusionmat(T_test,c-1,'order',[0,1,2,3,4]); % confusion matrix