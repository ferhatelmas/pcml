function test_err = llse_classifier(X_L, X_R, T, X_L_test, X_R_test, T_test)
%LLSE_CLASSIFIER(X_L, X_R, T, X_L_test, X_R_test, T_test): linear classification with least squares Tikhonov
%regularizer

% concatenate left and right camera, absorb bias into weight vector by
% adding a 1 to the end
[d,n] = size(X_L);
[~,n_test] = size(X_L_test);

X = [X_L; X_R; ones(1,n)]';
X_test = [X_L_test; X_R_test; ones(1,n_test)]';

T_T = encoder(T);
T_T_test = encoder(T_test);

M = 10; % cross validation fold #
v = 0:19; % set of possible regularizer parameter values

% cross validation
[bias_avg variance_avg] = cross_validation(X,T_T,v,M);
% minimum of the difference is where optimum v is located at
[~,ind] = min(abs(bias_avg - variance_avg));
v_opt = v(ind);
   
% solve for optimum weight vector using optimum v
A = X'*X + v_opt*eye(d);
B = X'*T_T;
R = chol(A); % cholesky decomposition of A where R'*R=A
sol1 = R'\B;
W = R\sol1;

test_err = regerr(X_test,W, T_T_test, v_opt);


