function [bias_avg variance_avg] = cross_validation(X, T_T, v, M)
%cross_validation(X, T_T, v, M)
% M-fold cross validation to pick v
% X: concatenated input matrix
% T_T: 1-M encoded target matrix
% v: set of Tikhonov constants
% M: fold #

% 10-fold cross validation to pick v
tn = length(v); % number of trials
for j=1:tn % runs for regularization parameters
    v_cur = v(j); 
    % hold averages for each trial
    bias_avg = zeros(1,tn);
    variance_avg = zeros(1,tn);
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
        A = X_tr'*X_tr + v_cur*eye(d);
        B = X_tr'*T_tr;
        W = A\B;
        
        % test performance on training fold (bias), accumulate for average
        bias = regerr(X_tr, W, T_tr, v_cur);
        bias_avg(j) = bias_avg(j) + bias;
        
        % test performance on validation fold (variance), accumulate for
        % average
        variance = regerr(X_val, W, T_val, v_cur);
        variance_avg(j) = variance_avg(j) + variance;
    end
    % calculate average over 10 trials
    bias_avg(j) = bias_avg(j)/M;
    variance_avg(j) = variance_avg(j)/M;
end


