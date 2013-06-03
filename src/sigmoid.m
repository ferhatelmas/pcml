function y = sigmoid(x)
%SIGMOID(x)
% sigmoid function
% x: input (scalar, vector or matrix)
% y: output (scalar, vector of matrix)

y = 1.0./(1.0+exp(-x));