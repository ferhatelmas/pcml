function [w_l_1, w_r_1, b_l_1, b_r_1, w_l_2, w_r_2, w_lr_2, b_l_2, b_r_2, w_3, b_3] = mlp_batch(X_L, X_R, T, X_L_v, X_R_v, T_v, h1, h2, nu, mu)
%MLP(X_L, X_R, T, h1, h2, nu, mu)
% X_L: Left input matrix
% X_R: Right input matrix
% h1: Number of neurons in the first layer
% h2: Number of neurons in the second layer
% nu: Learning rate
% mu: Momentum term

% dimensions of input vectors
% d - dimension of the input space
% n - number of the inputs = batch size (not needed)
d = size(X_L,1);

% layer 1 - weight initializations
w_l_1 = initialize_weights(h1, d);
w_r_1 = initialize_weights(h1, d);
b_l_1 = initialize_weights(h1, 1);
b_r_1 = initialize_weights(h1, 1);

% layer 2 - weight initializations
w_l_2 = initialize_weights(h2, h1);
w_r_2 = initialize_weights(h2, h1);
w_lr_2 = initialize_weights(h2, 2*h1);
b_l_2 = initialize_weights(h2, 1);
b_r_2 = initialize_weights(h2, 1);
b_lr_2 = initialize_weights(h2, 1);

% layer 3 - weight initializations
w_3 = initialize_weights(1, h2);
b_3 = initialize_weights(1, 1);

% initialize history weight matrices and bias vectors
hw_l_1 = w_l_1;
hw_r_1 = w_r_1;
hw_l_2 = w_l_2;
hw_r_2 = w_r_2;
hw_lr_2 = w_lr_2;
hw_3 = w_3;
hb_l_1 = b_l_1;
hb_r_1 = b_r_1;
hb_l_2 = b_l_2;
hb_r_2 = b_r_2;
hb_lr_2 = b_lr_2;
hb_3 = b_3;

% the following goes inside a while loop that checks the validation error
% and stops the update when it starts increasing

val_err = 0; % current validation error
val_err_prev = 0; % previous validation error
tn = 0; % trial number

while(val_err - val_err_prev <= 0) % difference is positive if val_err is increasing

% do forward pass
[a_l_1, a_r_1, z_l_1, z_r_1, a_l_2, a_r_2, a_lr_2, z_2, a_3] = mlp_forward(X_L, X_R, ...
  w_l_1, b_l_1, w_r_1, b_r_1, ...
  w_l_2, b_l_2, w_r_2, b_r_2, w_lr_2, b_lr_2, ...
  w_3, b_3);

% do backward pass
[dw_l_1, dw_r_1, dw_l_2, dw_r_2, dw_lr_2, dw_3, r_l_1, r_r_1, r_l_2, r_r_2, r_lr_2, r_3] = mlp_backward(X_L, X_R, T, ...
  a_l_1, a_r_1, ...
  a_l_2, a_r_2, a_lr_2, ...
  a_3, ...
  z_l_1, z_r_1, ...
  z_2, ...
  w_l_2, w_r_2, w_lr_2, w_3);

temp = gradient_descent(hw_l_1, w_l_1, dw_l_1, nu, mu);
hw_l_1 = w_l_1; w_l_1 = temp;

temp = gradient_descent(hw_r_1, w_r_1, dw_r_1, nu, mu);
hw_r_1 = w_r_1; w_r_1 = temp;

temp = gradient_descent(hw_l_2, w_l_2, dw_l_2, nu, mu);
hw_l_2 = w_l_2; w_l_2 = temp;

temp = gradient_descent(hw_r_2, w_r_2, dw_r_2, nu, mu);
hw_r_2 = w_r_2; w_r_2 = temp;

temp = gradient_descent(hw_lr_2, w_lr_2, dw_lr_2, nu, mu);
hw_lr_2 = w_lr_2; w_lr_2 = temp;

temp = gradient_descent(hw_3, w_3, dw_3, nu, mu);
hw_3 = w_3; w_3 = temp;

% update biases
temp = gradient_descent(hb_l_1, b_l_1, r_l_1, nu, mu);
hb_l_1 = b_l_1; b_l_1 = temp;

temp = gradient_descent(hb_r_1, b_r_1, r_r_1, nu, mu);
hb_r_1 = b_r_1; b_r_1 = temp;

temp = gradient_descent(hb_l_2, b_l_2, r_l_2, nu, mu);
hb_l_2 = b_l_2; b_l_2 = temp;

temp = gradient_descent(hb_r_2, b_r_2, r_r_2, nu, mu);
hb_r_2 = b_r_2; b_r_2 = temp;

temp = gradient_descent(hb_lr_2, b_lr_2, r_lr_2, nu, mu);
hb_lr_2 = b_lr_2; b_lr_2 = temp;

temp = gradient_descent(hb_3, b_3, r_3, nu, mu);
hb_3 = b_3; b_3 = temp;

% calculate training and updated validation error

tr_err = logerr(T,a_3);

% do forward pass
[~, ~, ~, ~, ~, ~, ~, ~, a_3] = mlp_forward(X_L_val, X_R_val, ...
  w_l_1, b_l_1, w_r_1, b_r_1, ...
  w_l_2, b_l_2, w_r_2, b_r_2, w_lr_2, b_lr_2, ...
  w_3, b_3);

val_err_prev = val_err;
val_err = logerr(T_v,a_3);

% visualize errors
tn = tn + 1;
plotter(tr_err, val_err, tn);

end
