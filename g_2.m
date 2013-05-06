function y = g_2(a_l,a_r,a_lr)
%G_2(x)
% second layer transfer function
% a_l, a_r, a_lr: inputs (scalar, vector or matrix)
% y: output (scalar, vector or matrix)

y = a_lr.*sigmoid(a_l).*sigmoid(a_r);