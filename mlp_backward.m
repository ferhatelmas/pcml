function [dw_l_1, dw_r_1, dw_l_2, dw_r_2, dw_lr_2, dw_3, db_l_1, db_r_1, db_l_2, db_r_2, db_lr_2, db_3] = mlp_backward(a_l_1, a_r_1, a_l_2, a_r_2, a_lr_2, a_3)
%MLP_BACKWARD(a_l_1, a_r_1, a_l_2, a_r_2, a_lr_2, a_3, b_l_1, b_r_1, b_l_2, b_r_2, b_lr_2, b_3)
% Calculates forward activation values for MLP
% a_l_1, a_r_1: first level activation vectors for left and right nodes respectively
% a_l_2, a_r_2, a_lr_2: second level activation vectors for left, right and
% left-right nodes respectively
% a_3: third layer activation output
% dw_l_1, dw_r_1: first level left and right gradient weight matrices respectively 
% dw_l_2, dw_r_2, dw_lr_2: second level left, right and left_right
% gradient weight matrices respectively
% dw_3: third level gradient weight vector
% db_l_1, db_r_1: first level left and right gradient bias vectors respectively 
% db_l_2, db_r_2, db_lr_2: second level left, right and left_right
% gradient bias vectors respectively
% db_3: third level gradient bias value
