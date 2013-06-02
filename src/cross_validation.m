function v_opt = cross_validation(X, T_T, v, M)
%CROSS_VALIDATION(X, T_T, v, M)
% M-fold cross validation to pick v
% X: concatenated input matrix
% T_T: 1-M encoded target matrix
% v: set of Tikhonov constants
% M: fold #

% n-fold cross validation to pick v
tn = length(v); % number of trials
[n,d] = size(X);

% initialize averages for each trial
tr_err_avg = zeros(1,tn);
val_err_avg = zeros(1,tn);
std_devs = zeros(1,tn);
val_errs = zeros(1,M); % holds errors for each fold to calculate std_dev

for j=1:tn % runs for regularization parameters
    
    v_cur = v(j); 
    c = n/M; % step size
    fold = 0; % fold number
    
    for i=0:c:n-1 % runs for validation folds
        
        fold = fold + 1;
        % build training and validation folds
        [X_tr, X_val] = divide(X, i, c);
        [T_tr, T_val] = divide(T_T, i, c);
        
        % normalize folds
        [m,istd] = find_parameters(X_tr');
        X_tr = [normalize(X_tr',m,istd)' ones(size(X_tr,1),1)];
        X_val = [normalize(X_val',m,istd)' ones(size(X_val,1),1)];
        
        % solve for optimum weight vector with training fold
        A = X_tr'*X_tr + v_cur*eye(d+1);
        B = X_tr'*T_tr;
        W = A\B;

        % test performance on validation fold (variance)
        % accumulate training and validation for average
        tr_err = squared_error(T_tr',(X_tr*W)');
        val_err = squared_error(T_val',(X_val*W)');
        tr_err_avg(j) = tr_err_avg(j) + tr_err;
        val_err_avg(j) = val_err_avg(j) + val_err;
        val_errs(fold) = val_err;
    end
    % calculate average and standard deviation over M trials  
    tr_err_avg(j) = tr_err_avg(j)/M;
    val_err_avg(j) = val_err_avg(j)/M;
    std_devs(j) = std(val_errs);
end

% minimum of validation error is where optimum v is located at
[~,ind] = min(val_err_avg);
v_opt = v(ind);

% visualize results
plotter_linear(v, tr_err_avg, val_err_avg, std_devs, v_opt, ind)
end

function [tr, val] = divide(m, i, c)
%DIVIDE(m, i, c)
% divides matrix m to two parts as training and validation
% m: matrix to be divided
% i: start index of fold
% c: # of elements inside of fold

tr = m; % back-up X, not to destroy during cross validation
val = tr(i+1:i+c, :);
tr(i+1:i+c, :) = [];
end
