function x = mlp(X_L, X_R, T, h1, h2, nu, mu, batch_size)
%MLP(X_L, X_R, T, h1, h2, nu, mu, batch_size)
% X_L: Left input matrix
% X_R: Right input matrix
% T: Correct class values for corresponding inputs
% h1: Number of neurons in the first layer
% h2: Number of neurons in the second layer
% nu: Learning rate
% mu: Momentum term
% batch_size: Batch size to be used in training (optional - unless given, default is 1) 

% dimensions of input vectors
% d - dimension of the input space
% n - number of the inputs
[d, n] = size(X_L);

% set default batch size
if nargin < 9
    batch_size = 1;
elseif mod(n, batch_size) ~= 0
    sprintf('Number of inputs(%d) is not multiples of batch size(%d)', n, d)
    return
end

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
hw_l_1 = w_l_1
hw_r_1 = w_r_1
hw_l_2 = w_l_2
hw_r_2 = w_r_2
hw_lr_2 = w_lr_2
hw_3 = w_3
hb_l_1 = b_l_1
hb_r_1 = b_r_1
hb_l_2 = b_l_2
hb_r_2 = b_r_2
hb_lr_2 = b_lr_2
hb_3 = b_3

% gradient sums 
reset_gradient_sums

% go over each input
for i=1:n
    % pick current inputs
    x_l = X_L(:, i);
    x_r = X_R(:, i);

    % pick correct output for corresponding inputs
    t = T(i)

    % do forward pass
    forward_results = mlp_forward(x_l, x_r, ...
      w_l_1, b_l_1, w_r_1, b_r_1, ...
      w_l_2, b_l_2, w_r_2, b_r_2, w_lr_2, b_lr_2, ...
      w_3, b_3);

    % do backward pass
    backward_results = mlp_backward(x_l, x_r, t, ...
      forward_results.a_l_1, forward_results.a_r_1, ...
      forward_results.a_l_2, forward_results.a_r_2, forward_results.a_lr_2, ...
      forward_results.a_3, ...
      forward_results.z_l_1, forward_results.z_r_1, ...
      forward_results.z_2, ...
      w_l_2, w_r_2, w_lr_2, w_3);

    % add new gradients generated by current input
    sdw_l_1  = sdw_l_1 + backward_results.dw_l_1;
    sdw_r_1  = sdw_r_1 + backward_results.dw_r_1;
    sdw_l_2  = sdw_l_2 + backward_results.dw_l_2;
    sdw_r_2  = sdw_r_2 + backward_results.dw_r_2;
    sdw_lr_2 = sdw_lr_2 + backward_results.dw_lr_2;
    sdw_3    = sdw_3 + backward_results.dw_3;

    sr_l_1 = sr_l_1 + backward_results.r_l_1;
    sr_r_1 = sr_r_1 + backward_results.r_r_1;
    sr_l_2 = sr_l_2 + backward_results.r_l_2;
    sr_r_2 = sr_r_2 + backward_results.r_r_2;
    sr_lr_2 = sr_lr_2 + backward_results.r_lr_2;
    sr_3 = sr_3 + backward_results.r_3;

    % if the number of processed inputs equals to batch_size
    % update weights with total gradients
    if mod(i, batch_size) == 0
        % update weights
        temp = gradient_descent(hw_l_1, w_l_1, sdw_l_1, nu, mu)
        hw_l_1 = w_l_1; w_l_1 = temp;
        
        temp = gradient_descent(hw_r_1, w_r_1, sdw_r_1, nu, mu)
        hw_r_1 = w_r_1; w_r_1 = temp;

        temp = gradient_descent(hw_l_2, w_l_2, sdw_l_2, nu, mu)
        hw_l_2 = w_l_2; w_l_2 = temp;

        temp = gradient_descent(hw_r_2, w_r_2, sdw_r_2, nu, mu)
        hw_r_2 = w_r_2; w_r_2 = temp;

        temp = gradient_descent(hw_lr_1, w_lr_1, sdw_lr_2, nu, mu)
        hw_lr_2 = w_lr_2; w_lr_2 = temp;

        temp = gradient_descent(hw_3, w_3, sdw_3, nu, mu)
        hw_3 = w_3; w_3 = temp;

        % update biases
        temp = gradient_descent(hb_l_1, b_l_1, sr_l_1, nu, mu)
        hb_l_1 = b_l_1; b_l_1 = temp;

        temp = gradient_descent(hb_r_1, b_r_1, sr_r_1, nu, mu)
        hb_r_1 = b_r_1; b_r_1 = temp;

        temp = gradient_descent(hb_l_2, b_l_2, sr_l_2, nu, mu)
        hb_l_2 = b_l_2; b_l_2 = temp;

        temp = gradient_descent(hb_r_2, b_r_2, sr_r_2, nu, mu)
        hb_r_2 = b_r_2; b_r_2 = temp;

        temp = gradient_descent(hb_lr_2, b_lr_2, sr_lr_2, nu, mu)
        hb_lr_2 = b_lr_2; b_lr_2 = temp;

        temp = gradient_descent(hb_3, b_3, sr_3, nu, mu)
        hb_3 = b_3; b_3 = temp;

        reset_gradient_sums
    end
end

    function reset_gradient_sums 
        % gradient sums 
        sdw_l_1 = zeros(size(w_l_1))
        sdw_r_1 = zeros(size(w_r_1))
        sdw_l_2 = zeros(size(w_l_2))
        sdw_r_2 = zeros(size(w_r_2))
        sdw_lr_2 = zeros(size(w_lr_2))
        sdw_3 = zeros(size(w_3))
        sr_l_1 = zeros(size(b_l_1))
        sr_r_1 = zeros(size(b_r_1))
        sr_l_2 = zeros(size(b_l_2))
        sr_r_2 = zeros(size(b_r_2))
        sr_lr_2 = zeros(size(b_lr_2))
        sr_3 = zeros(size(b_3))
    end 
end