function y = sigmoid_der(x)
%SIGMOID_DER(x)
% derivative of sigmoid function (1-sigmoid(x))
% x: input (scalar, vector or matrix)
% y: output (scalar, vector of matrix)

y = 1.0./(1.0+exp(x));
