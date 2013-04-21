function y = sigmoid(x)
%SIGMOID(x)
% sigmoid function
% x: input (scalar or vector)
% y: output (scalar or vector)

y = 1.0./(1.0+exp(-x));