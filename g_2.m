function y = g_2(a_l,a_r,a_lr)
%G_2(x)
% second layer transfer function
% a_l, a_r, a_lr: inputs (scalar or vector)
% y: output (scalar or vector)

y = a_lr.*sigmoid(a_l).*sigmoid(a_r);