function [tr_err, val_err, zero_one_error, logistic_error_test, zero_one_error_test min_i max_i] = mlp_binary(h1, h2, nu, mu, batch_size)
%MLP_BINARY(h1, h2, nu, mu, batch_size)
% h1: Number of neurons in the first layer
% h2: Number of neurons in the second layer
% nu: Learning rate
% mu: Momentum term
% batch_size: Number of input to be processed in one batch

% Gets data from base workspace
X_L = evalin('base', 'X_L');
X_R = evalin('base', 'X_R');
X_L_val = evalin('base', 'X_L_val');
X_R_val = evalin('base', 'X_R_val');
X_L_test = evalin('base', 'X_L_test');
X_R_test = evalin('base', 'X_R_test');
T = evalin('base', 'T');
T_val = evalin('base', 'T_val');
T_test = evalin('base', 'T_test');

% Runs mlp_batch to get optimum weights and biases
[w_l_1, w_r_1, b_l_1, b_r_1, ...
 w_l_2, w_r_2, w_lr_2, b_l_2, b_r_2, b_lr_2, ...
 w_3, b_3, tr_err, val_err, zero_one_error] = mlp_batch(X_L, X_R, T, X_L_val, X_R_val, T_val, h1, h2, nu, mu, batch_size);

% Does a forward pass to predict class labels
[~, ~, ~, ~, ~, ~, ~, ~, a_3] = mlp_forward(X_L_test, X_R_test, ...
                                            w_l_1, b_l_1, w_r_1, b_r_1, ...
                                            w_l_2, b_l_2, w_r_2, b_r_2, w_lr_2, b_lr_2, ...
                                            w_3, b_3);
   
% Calculates logistic and zero error for the predicted class labels                                        
logistic_error_test = logerr(T_test, a_3);
patterns = T_test.*a_3;
zero_one_error_test = mean(patterns < 0);
errored = find(patterns<0); % indices of misclassified patterns
patterns(patterns >= 0) = [];
[~, max_i] = max(patterns); % almost correctly classified
[~, min_i] = min(patterns); % very wrongly classified
% real indices of the two
max_i = errored(max_i);
min_i = errored(min_i);

end