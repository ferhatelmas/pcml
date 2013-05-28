function test_err = llse_classifier(X, T, X_test, T_test)
%LLSE_CLASSIFIER(X_L, X_R, T, X_L_test, X_R_test, T_test): linear classification with least squares Tikhonov
%regularizer

% concatenate left and right camera, absorb bias into weight vector by
% adding a 1 to the end
[~,n] = size(X);
[~,n_test] = size(X_test);

X = [X; ones(1,n)]';
X_test = [X_test; ones(1,n_test)]';

T_T = encoder(T);
T_T_test = encoder(T_test);

M = 10; % cross validation fold #
v = 10; % set of possible regularizer parameter values

% cross validation
val_err_avg = cross_validation(X(:,1:end-1),T_T,v,M);
% minimum of validation error is where optimum v is located at
[~,ind] = min(val_err_avg);
v_opt = v(ind);
   
% solve for optimum weight vector using optimum v
A = X'*X + v_opt*eye(size(X,2));
B = X'*T_T;
W = A\B;

% normalize test set with parameters of training set
test_err = regerr(X_test,W, T_T_test, v_opt);