function y = sigmoid_der(x)
%SIGMOID_DER(x)
% derivative of sigmoid function sigmoid(x)*(1-sigmoid(x))
% x: input (scalar, vector or matrix)
% y: output (scalar, vector of matrix)


y = sigmoid(x) ./ (1.0+exp(x));
