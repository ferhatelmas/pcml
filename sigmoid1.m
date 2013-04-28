function y = sigmoid1(x)
%SIGMOID1(x)
% sigmoid function
% x: input (scalar or vector)
% y: output (scalar or vector)

y = 1.0/(1.0+exp(x));
