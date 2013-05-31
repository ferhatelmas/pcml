function [hw_l_1, hw_r_1, hb_l_1, hb_r_1, hw_l_2, hw_r_2, hw_lr_2, hb_l_2, hb_r_2, hb_lr_2, hw_3, hb_3, tr_err, val_err, zero_one_error] = mlp_batch(X_L, X_R, T, X_L_val, X_R_val, T_val, h1, h2, nu, mu, batch_size)
%MLP_BATCH(X_L, X_R, T, X_L_val, X_R_val, T_val, h1, h2, nu, mu, batch_size)
% X_L: Left input matrix
% X_R: Right input matrix
% T: Vector of classes of training inputs
% X_L_val: Left validation input matrix
% X_R_val: Right validation input matrix
% T_val: Vector of classes of validation inputs
% h1: Number of neurons in the first layer
% h2: Number of neurons in the second layer
% nu: Learning rate
% mu: Momentum term
% batch_size: Number of input to be processed in one batch

% dimensions of input vectors
% d - dimension of the input space
% n - number of the inputs
[d, n] = size(X_L);

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
hw_l_1 = w_l_1; hw_r_1 = w_r_1;
hw_l_2 = w_l_2; hw_r_2 = w_r_2; hw_lr_2 = w_lr_2;
hw_3 = w_3;

hb_l_1 = b_l_1; hb_r_1 = b_r_1;
hb_l_2 = b_l_2; hb_r_2 = b_r_2; hb_lr_2 = b_lr_2;
hb_3 = b_3;

% randomize columns of training data to change ordering
randp = randperm(n);
X_L = X_L(:,randp);
X_R = X_R(:,randp);
T = T(randp);

ec = 0; % epoch count
early_stop = ones(1,2); % stores last two differences btw consecutive zero_one errors
es = 0;
prev_err = 0;

% initialize vectors to accumulate errors after each epoch
max_ec = 50;
tr_err = zeros(1,max_ec);
val_err = zeros(1,max_ec);
zero_one_error = zeros(1,max_ec);

while(sum(early_stop > 1e-6*ones(1,2)) > 0) 
    es = mod(es + 1,2);
    ec = ec+1;
    % process one batch of the inputs
    for i=1:batch_size:n
        % upper index to slice input matrices
        j = i + batch_size - 1;
        % one batch from corresponding inputs 
        x_l = X_L(:, i:j);
        x_r = X_R(:, i:j);
        t = T(i:j);

        % do a forward pass
        [a_l_1, a_r_1, ...
         z_l_1, z_r_1, ...
         a_l_2, a_r_2, a_lr_2, ...
         z_2, a_3] = mlp_forward(x_l, x_r, ...
                                 w_l_1, b_l_1, w_r_1, b_r_1, ...
                                 w_l_2, b_l_2, w_r_2, b_r_2, w_lr_2, b_lr_2, ...
                                 w_3, b_3);

        % do a backward pass
        [dw_l_1, dw_r_1, ...
         dw_l_2, dw_r_2, dw_lr_2, ...
         dw_3, ...
         r_l_1, r_r_1, ...
         r_l_2, r_r_2, r_lr_2, ...
         r_3] = mlp_backward(x_l, x_r, t, ...
                             a_l_1, a_r_1, ...
                             a_l_2, a_r_2, a_lr_2, ...
                             a_3, ...
                             z_l_1, z_r_1, ...
                             z_2, ...
                             w_l_2, w_r_2, w_lr_2, w_3);


        % update weight matrices
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

        % update bias vectors
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
        
    end

% do one more forward pass to get labels for these epoch
[~, ~, ~, ~, ~, ~, ~, ~, a_3] = mlp_forward(X_L, X_R, ...
                                            w_l_1, b_l_1, w_r_1, b_r_1, ...
                                            w_l_2, b_l_2, w_r_2, b_r_2, w_lr_2, b_lr_2, ...
                                            w_3, b_3);

% calculate training error
tr_err(ec) = logerr(T,a_3);


% do a forward pass to get updated class labels
[~, ~, ~, ~, ~, ~, ~, ~, a_3] = mlp_forward(X_L_val, X_R_val, ...
                                            w_l_1, b_l_1, w_r_1, b_r_1, ...
                                            w_l_2, b_l_2, w_r_2, b_r_2, w_lr_2, b_lr_2, ...
                                            w_3, b_3);
% update validation error
% val_err_prev = val_err;
val_err(ec) = logerr(T_val,a_3)

% calculate 0-1 error for validation set
zero_one_error(ec) = mean(T_val.*a_3 < 0);
early_stop(es+1) = abs(zero_one_err(ec) - prev_err);
prev_err = zero_one_err(ec);

% visualize errors
plotter(tr_err, val_err, zero_one_error, ec);

end