function [w_l_1, w_r_1, b_l_1, b_r_1, w_l_2, w_r_2, w_lr_2, b_l_2, b_r_2, b_lr_2, w_3, b_3, tr_err] = mlp_epoch(X_L, X_R, T, h1, h2, nu, mu, batch_size)
%MLP_EPOCH(X_L, X_R, T, h1, h2, nu, mu, batch_size): runs one epoch of MLP
%training
% X_L: Left input matrix
% X_R: Right input matrix
% T: Vector of classes of training inputs
% h1: Number of neurons in the first layer
% h2: Number of neurons in the second layer
% nu: Learning rate
% mu: Momentum term
% batch_size: Number of input to be processed in one batch

format long

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

% initialize vector to store MLP outputs
a_3_v = ones(1,n);
% randomize columns of training data to change ordering
X_L = X_L(:,randperm(n));
X_R = X_R(:,randperm(n));

%while(val_err - val_err_prev <= 0)  % difference is positive if val_err is increasing

    % process epoch of inputs
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

        a_3_v(i) = a_3;
    end

% calculate training error
tr_err = logerr(T,a_3_v);

end