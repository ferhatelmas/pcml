function y = sigmoid_der(x)
%SIGMOID_DER(x)
% derivative of sigmoid function (1-sigmoid(x))
% x: input (scalar or vector)
% y: output (scalar or vector)

y = 1.0/(1.0+exp(x));
