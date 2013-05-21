function [logistic_error, zero_one_error] = mlp_binary(w_l_1, b_l_1, w_r_1, b_r_1, w_l_2, b_l_2, w_r_2, b_r_2, w_lr_2, b_lr_2, w_3, b_3)
%MLP_BINARY(w_l_1, b_l_1, w_r_1, b_r_1, w_l_2, b_l_2, w_r_2, b_r_2, w_lr_2, b_lr_2, w_3, b_3)
% Runs binary mlp with the given weights and bias
% (w_l_1, b_l_1), (w_r_1, b_r_1): first level initial weight matrices and bias vectors
% (w_l_2, b_l_2), (w_r_2, b_r_2), (w_lr_2, b_lr_2): second level initial weight matrices and bias vectors
% (w_3, b_3): last level weight matrice and bias scalar

% Imports data into base workspace
importfile('../data/norb_binary.mat');

% Removes training data from base workspace
clear GLOBAL train_cat_s train_left_s train_right_s

% Gets test data into current workspace
global test_cat_s test_left_s test_right_s

% Maps class labels from {1, 3} to {-1, 1}
test_cat_s = test_cat_s - 2;

% Does a forward pass to predict class labels
[~, ~, ~, ~, ~, ~, ~, ~, a_3] = mlp_forward(test_left_s, test_right_s, ...
                                            w_l_1, b_l_1, w_r_1, b_r_1, ...
                                            w_l_2, b_l_2, w_r_2, b_r_2, w_lr_2, b_lr_2, ...
                                            w_3, b_3);
   
% Calculates logictic and zero error for the predicted class labels                                        
logistic_error = logerr(test_cat_s, a_3);
zero_one_error = mean(T.*a_3 < 0);

% Removes test data from base workspace
clear GLOBAL test_cat_s test_left_s test_right_s