function test_err = linear_regression(X, T, X_test, T_test)
%LINEAR_REGRESSION(X, T, X_test, T_test)
% linear regression with squared error
% CALL PREPARE_LINEAR BEFORE RUNNING

% get number of cases
[~,n] = size(X);
[~,n_test] = size(X_test);

% adding a 1 to the end as bias
X = [X; ones(1,n)]';
X_test = [X_test; ones(1,n_test)]';

T_T = encoder(T);
T_T_test = encoder(T_test);
   
% solve for optimum weight vector
A = X'*X;
B = X'*T_T;
W = A\B;

% calculate test error for optimum W
test_err = squared_error(T_T_test', (X_test*W)');