function [tr_err, val_err, zero_one_error, sq_error_test, confused] = mlp_multi(h1, h2, nu, mu, batch_size)
%MLP_MULTI(h1, h2, nu, mu, batch_size)
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

% Runs mlp_train_multi to get optimum weights and biases
[w_l_1, w_r_1, b_l_1, b_r_1, ...
 w_l_2, w_r_2, w_lr_2, b_l_2, b_r_2, b_lr_2, ...
 w_3, b_3, tr_err, val_err, zero_one_error] = mlp_train_multi(X_L, X_R, T, X_L_val, X_R_val, T_val, h1, h2, nu, mu, batch_size);

% Does a forward pass to predict class labels
[~, ~, ~, ~, ~, ~, ~, ~, a_3] = mlp_forward(X_L_test, X_R_test, ...
                                            w_l_1, b_l_1, w_r_1, b_r_1, ...
                                            w_l_2, b_l_2, w_r_2, b_r_2, w_lr_2, b_lr_2, ...
                                            w_3, b_3);
   
% Calculates squared error for the predicted class labels                                        
sq_error_test = sqrerr(encoder(T_test)', a_3);
[~,c] = max(a_3,[],1); % find index of maximum among each sample output
confused = confusionmat(T_test,c-1,'order',[0,1,2,3,4]); % confusion matrix

end