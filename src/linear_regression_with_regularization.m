function [test_err, zero_one_err, confused] = linear_regression_with_regularization(X, T, X_test, T_test)
%LINEAR_REGRESSION_WITH_REGULARIZATION(X, T, X_test, T_test)
% linear regression with Tikhonov regularizer
% CALL PREPARE_LINEAR BEFORE RUNNING

% get number of cases
[~,n] = size(X);
[~,n_test] = size(X_test);

% adding a 1 to the end as bias
[m, istd] = find_parameters(X);
X = [X; ones(1,n)]';
X_test = [X_test; ones(1,n_test)]';

T_T = encoder(T);
T_T_test = encoder(T_test);

M = 10; % cross validation fold #
c = 1e-16;
v = [0 c*10.^(0:19)]; % set of possible regularizer parameter values

% cross validation
v_opt = cross_validation(X(:,1:end-1),T_T,v,M);

% solve for optimum weight vector using optimum v
X = [normalize(X(:,1:end-1)',m,istd) ; ones(1,n)]'; 
A = X'*X + v_opt*eye(size(X,2));
B = X'*T_T;
W = A\B;

% calculate test error for optimum W
test_err = regularized_error(X_test, W, T_T_test, v_opt);
% zero-one error
Y = X_test*W;
[~,c] = max(Y,[],2); % find index of maximum among each sample output
zero_one_err = mean((c-1)' ~= T_test);
confused = confusionmat(T_test,(c-1)','order',[0,1,2,3,4]); % confusion matrix
