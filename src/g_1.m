function y = g_1(x)
%G_1(x)
% first layer transfer function
% x: input (scalar, vector or matrix)
% y: output (scalar, vector or matrix)

y = 2*sigmoid(2.*x)-1;