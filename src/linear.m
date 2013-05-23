function W = linear(X_L, X_R, t)
%linear(X_L, X_R, T): linear classification with least squares Tikhonov
%regularizer

k = 5; % class size
M = 10; % cross validation fold #
v = 0:19; % set of possible regularizer parameter values

% concatenate left and right camera, absorb bias into weight vector by
% adding a 1 to the end
[d,n] = size(X_L);
X = [X_L; X_R; ones(1,n)]';

T_T = encoder(t,k)';

% 10-fold cross validation to pick v
for j=1:length(v) % runs for regularization parameters
    v_cur = v(j); 
    for i=0:M:n % runs for validation folds
        X_cv = X; % back-up X, not to destroy during cross validation
        X_val = X_cv(i+1:i+M,:);
        X_cv(i+1:i+M,:) = [];
        X_tr = X_cv;
        T_cv = T_T;
        T_val = T_cv(i+1:i+M,:);
        T_cv(i+1:i+M,:) = [];
        T_tr = T_cv;
        
        % solve for optimum weight vector with training fold
        A = X_tr'*X_tr + v*eye(d);
        B = X_tr'*T_tr;
        W = A\B;
        
        % test performance on validation fold
        ...
    end
% calculate average over 10 trials, pick v as optimum if error is minimum
...
end
    
   
% solve for optimum weight vector using optimum v
A = X'*X + v_opt*eye(d);
B = X'*T_T;
W = A\B;


