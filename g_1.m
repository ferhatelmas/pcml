function y = g_1(x)
%G_1(x)
% first layer transfer function
% x: input (scalar or vector)
% y: output (scalar or vector)

y = 2*sigmoid(2.*x)-1;