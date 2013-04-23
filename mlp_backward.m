function [dw_l_1, dw_r_1, dw_l_2, dw_r_2, dw_lr_2, dw_3, r_l_1, r_r_1, r_l_2, r_r_2, r_lr_2, r_3] = mlp_backward(x_l, x_r, t, a_l_1, a_r_1, a_l_2, a_r_2, a_lr_2, a_3, z_l_1, z_r_1, z_2, w_l_2, w_r_2, w_lr_2, w_3)
%MLP_BACKWARD(a_l_1, a_r_1, a_l_2, a_r_2, a_lr_2, a_3, b_l_1, b_r_1, b_l_2, b_r_2, b_lr_2, b_3)
% Calculates gradients for weight and bias vectors using backpropogation.
% x_l, x_r: left and right input vectors
% t: assigned class value
% a_l_1, a_r_1: first level activation vectors for left and right nodes respectively
% a_l_2, a_r_2, a_lr_2: second level activation vectors for left, right and
% left-right nodes respectively
% a_3: third layer activation output
% z_l_1, z_r_1, z_2: non-linear transformations of activation vectors
% w_l_2, w_r_2, w_lr_2, w_3: weight matrices to be updated
% 
% dw_l_1, dw_r_1: first level left and right gradient weight matrices respectively 
% dw_l_2, dw_r_2, dw_lr_2: second level left, right and left_right
% gradient weight matrices respectively
% dw_3: third level gradient weight vector
% db_l_1, db_r_1: first level left and right gradient bias vectors respectively 
% db_l_2, db_r_2, db_lr_2: second level left, right and left_right
% gradient bias vectors respectively
% db_3: third level gradient bias value

% build z_lr vector from z_l_1 and z_r_1
z_lr_1 = [z_l_1 ; z_r_1];

% linear transformation of t from [-1,1] to [0,1]
t_t = 0.5*(t+1);

% transfer function derivatives
g_1_d_l = 4*sigmoid(2*a_l_1).*(1-2*sigmoid(2*a_l_1));
g_1_d_r = 4*sigmoid(2*a_r_1).*(1-2*sigmoid(2*a_r_1));
g_2_d_l = a_lr_2.*sigmoid(a_r_2).*sigmoid(a_l_2).*(1-sigmoid(a_l_2));
g_2_d_r = a_lr_2.*sigmoid(a_l_2).*sigmoid(a_r_2).*(1-sigmoid(a_r_2));
g_2_d_lr = sigmoid(a_l_2).*sigmoid(a_r_2);

% residual calculations
r_3 = sigmoid(a_3) - t_t;
r_l_2 = diag(g_2_d_l)*w_3'*r_3;
r_r_2 = diag(g_2_d_r)*w_3'*r_3;
r_lr_2 = diag(g_2_d_lr)*w_3'*r_3;
r_l_1 = diag(g_1_d_l)*(w_l_2'*r_l_2 + (w_lr_2(:,1:h1))'*r_lr_2);
r_r_1 = diag(g_1_d_r)*(w_r_2'*r_r_2 + (w_lr_2(:,h1+1:end))'*r_lr_2);

% gradient calculations
dw_3 = r_3*z_2';
dw_l_2 = r_l_2*z_l_1';
dw_r_2 = r_r_2*z_r_1';
dw_lr_2 = r_lr_2*z_lr_1';
dw_l_1 = r_l_1*x_l';
dw_r_1 = r_r_1*x_r';