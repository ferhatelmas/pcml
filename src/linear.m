function output = linear(x_l, x_r, t)
%linear(x_l, x_r, v): linear classification with least squares Tikhonov
%regularizer

% concatenate left and right camera, absorb bias into weight vector by
% adding a 1 to the end
x = [x_l; x_r; 1];

d = length(x);
w = initialize_weights(x,1);

% obtain regularization parameter using 10-fold cross validation
v = cross_validation;

% solve for optimum weight vector
A = x'*x + v*eye(d)
B = x'*t;
w = inv(A)*B;

y = w'*x;

