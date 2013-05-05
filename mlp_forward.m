function results = mlp_forward(x_l, x_r, w_l_1, b_l_1, w_r_1, b_r_1, w_l_2, b_l_2, w_r_2, b_r_2, w_lr_2, b_lr_2, w_3, b_3)
%MLP_FORWARD 
% Implements forward pass of MLP
%  x_l, x_r: left and right input vectors
% (w_l_1, b_l_1), (w_r_1, b_r_1): first level initial weight matrices and bias vectors
% (w_l_2, b_l_2), (w_r_2, b_r_2), (w_lr_2, b_lr_2): second level initial weight matrices and bias vectors
% (w_3, b_3): last level weight matrice and bias scalar

% first level activation values
a_l_1 = w_l_1 * x_l + b_l_1;
a_r_1 = w_r_1 * x_r + b_r_1;

% first level non-linear transforms
z_l_1 = g_1(a_l_1);
z_r_1 = g_1(a_r_1);
z_lr_1 = [z_l_1; z_r_1];

% second level activation values
a_l_2 = w_l_2 * z_l_1 + b_l_2;
a_r_2 = w_r_2 * z_r_1 + b_r_2;
a_lr_2 = w_lr_2 * z_lr_1 + b_lr_2;

% second level non-linear tranforms
z_2 = g_2(a_l_2, a_r_2, a_lr_2);

% third level activation value
a_3 = w_3 * z_2 + b_3;

% make a structure to return all calculations
results = struct('a_l_1', a_l_1, 'a_r_1', a_r_1, 'z_l_1', z_l_1, 'z_r_1', z_r_1, ...
    'a_l_2', a_l_2, 'a_r_2', a_r_2, 'a_lr_2', a_lr_2, 'z_2', z_2, 'a_3', a_3)

end