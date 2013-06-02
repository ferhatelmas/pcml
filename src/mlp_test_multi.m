x = rand(25,1);

small_increment = 1e-6;

[d, n] = size(x);

h1 = 3; % first layer
h2 = 2; % second layer
t = 4;
t_t = encoder(t)';
k = 5;

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
w_3 = initialize_weights(k, h2);
b_3 = initialize_weights(k, 1);

for i=1:h1
    for j=1:d 
        % do a forward pass
        [~, ~, ~, ~, ~, ~, ~, ~, ...
         a_3] = mlp_forward(x, x, ...
                            w_l_1, b_l_1, w_r_1, b_r_1, ...
                            w_l_2, b_l_2, w_r_2, b_r_2, w_lr_2, b_lr_2, ...
                            w_3, b_3);
                       
        error_base = squared_error(t_t, a_3);
       
        % add small values to weights
        updated_w_l_1 = w_l_1;
        updated_w_l_1(i, j) = updated_w_l_1(i, j) + small_increment;
        
        updated_w_r_1 = w_r_1;
        updated_w_r_1(i, j) = updated_w_r_1(i, j) + small_increment;
        
        % do a forward pass
        [a_l_1, a_r_1, ...
         z_l_1, z_r_1, ...
         a_l_2, a_r_2, a_lr_2, ...
         z_2, a_3] = mlp_forward(x, x, ...
                                 updated_w_l_1, b_l_1, w_r_1, b_r_1, ...
                                 w_l_2, b_l_2, w_r_2, b_r_2, w_lr_2, b_lr_2, ...
                                 w_3, b_3);

        % do a backward pass
        [dw_l_1, ~, ~, ~, ~, ~, ...
         ~, ~, ~, ~, ~, ~] = mlp_backward_multi(x, x, t, ...
                                                a_l_1, a_r_1, ...
                                                a_l_2, a_r_2, a_lr_2, ...
                                                a_3, ...
                                                z_l_1, z_r_1, ...
                                                z_2, ...
                                                w_l_2, w_r_2, w_lr_2, w_3);

        error_left = squared_error(t_t, a_3);
        derivative_left = (error_left - error_base) / small_increment;
        diff_left = abs(derivative_left - dw_l_1(i, j));
         
        % do a forward pass
        [a_l_1, a_r_1, ...
         z_l_1, z_r_1, ...
         a_l_2, a_r_2, a_lr_2, ...
         z_2, a_3] = mlp_forward(x, x, ...
                                 w_l_1, b_l_1, updated_w_r_1, b_r_1, ...
                                 w_l_2, b_l_2, w_r_2, b_r_2, w_lr_2, b_lr_2, ...
                                 w_3, b_3);

        % do a backward pass
        [~, dw_r_1, ~, ~, ~, ~, ...
         ~, ~, ~, ~, ~, ~] = mlp_backward_multi(x, x, t, ...
                                                a_l_1, a_r_1, ...
                                                a_l_2, a_r_2, a_lr_2, ...
                                                a_3, ...
                                                z_l_1, z_r_1, ...
                                                z_2, ...
                                                w_l_2, w_r_2, w_lr_2, w_3);
                                  
        error_right = squared_error(t_t, a_3);
        derivative_right = (error_right - error_base) / small_increment;
        diff_right = abs(derivative_right - dw_r_1(i, j));
                                         
        if max(diff_left, diff_right) > small_increment
            disp('Test failed: difference is significant');
            sprintf('Left: %.12g, Right: %.12g', diff_left, diff_right)
            return
        end
    end
end

disp('Test is successful');