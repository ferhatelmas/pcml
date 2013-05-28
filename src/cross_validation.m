function val_err_avg = cross_validation(X, T_T, v, M)
%cross_validation(X, T_T, v, M)
% M-fold cross validation to pick v
% X: concatenated input matrix
% T_T: 1-M encoded target matrix
% v: set of Tikhonov constants
% M: fold #

% 10-fold cross validation to pick v
tn = length(v); % number of trials
[n,d] = size(X);
for j=1:tn % runs for regularization parameters
    v_cur = v(j); 
    % hold averages for each trial
    val_err_avg = zeros(1,tn);
    c = n/M;
    for i=0:c:n-1 % runs for validation folds
        disp(i);
        X_cv = X; % back-up X, not to destroy during cross validation
        X_val = X_cv(i+1:i+c,:);
        X_cv(i+1:i+c,:) = [];
        X_tr = X_cv;
        T_cv = T_T;
        T_val = T_cv(i+1:i+c,:);
        T_cv(i+1:i+c,:) = [];
        T_tr = T_cv;
        
        % normalize folds
        [m,istd] = find_par(X_tr');
        X_tr = normalize(X_tr',m,istd)';
        X_val = normalize(X_val',m,istd)';
        
        % solve for optimum weight vector with training fold
        A = X_tr'*X_tr + v_cur*eye(d);
        B = X_tr'*T_tr;
        W = A\B;

        % test performance on validation fold (variance), accumulate for
        % average
        variance = regerr(X_val, W, T_val, v_cur);
        val_err_avg(j) = val_err_avg(j) + variance;
    end
    % calculate average over 10 trials
    val_err_avg(j) = val_err_avg(j)/M;
end


